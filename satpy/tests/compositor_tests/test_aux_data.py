#!/usr/bin/env python
# Copyright (c) 2018-2025 Satpy developers
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

"""Tests for compositors using auxiliary data."""

import os
import unittest
from unittest import mock

import pytest

import satpy


class TestStaticImageCompositor(unittest.TestCase):
    """Test case for the static compositor."""

    @mock.patch("satpy.area.get_area_def")
    def test_init(self, get_area_def):
        """Test the initializiation of static compositor."""
        from satpy.composites.aux_data import StaticImageCompositor

        # No filename given raises ValueError
        with pytest.raises(ValueError, match="StaticImageCompositor needs a .*"):
            StaticImageCompositor("name")

        # No area defined
        comp = StaticImageCompositor("name", filename="/foo.tif")
        assert comp._cache_filename == "/foo.tif"
        assert comp.area is None

        # Area defined
        get_area_def.return_value = "bar"
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        assert comp._cache_filename == "/foo.tif"
        assert comp.area == "bar"
        get_area_def.assert_called_once_with("euro4")

    @mock.patch("satpy.aux_download.retrieve")
    @mock.patch("satpy.aux_download.register_file")
    @mock.patch("satpy.Scene")
    def test_call(self, Scene, register, retrieve):  # noqa
        """Test the static compositing."""
        from satpy.composites.aux_data import StaticImageCompositor

        satpy.config.set(data_dir=os.path.join(os.path.sep, "path", "to", "image"))
        remote_tif = "http://example.com/foo.tif"

        class MockScene(dict):
            def load(self, arg):
                pass

        img = mock.MagicMock()
        img.attrs = {}
        scn = MockScene()
        scn["image"] = img
        Scene.return_value = scn
        # absolute path to local file
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        res = comp()
        Scene.assert_called_once_with(reader="generic_image",
                                      filenames=["/foo.tif"])
        register.assert_not_called()
        retrieve.assert_not_called()
        assert res.attrs["sensor"] is None
        assert "modifiers" not in res.attrs
        assert "calibration" not in res.attrs

        # remote file with local cached version
        Scene.reset_mock()
        register.return_value = "data_dir/foo.tif"
        retrieve.return_value = "data_dir/foo.tif"
        comp = StaticImageCompositor("name", url=remote_tif, area="euro4")
        res = comp()
        Scene.assert_called_once_with(reader="generic_image",
                                      filenames=["data_dir/foo.tif"])
        assert res.attrs["sensor"] is None
        assert "modifiers" not in res.attrs
        assert "calibration" not in res.attrs

        # Non-georeferenced image, no area given
        img.attrs.pop("area")
        comp = StaticImageCompositor("name", filename="/foo.tif")
        with pytest.raises(AttributeError):
            comp()

        # Non-georeferenced image, area given
        comp = StaticImageCompositor("name", filename="/foo.tif", area="euro4")
        res = comp()
        assert res.attrs["area"].area_id == "euro4"

        # Filename contains environment variable
        os.environ["TEST_IMAGE_PATH"] = "/path/to/image"
        comp = StaticImageCompositor("name", filename="${TEST_IMAGE_PATH}/foo.tif", area="euro4")
        assert comp._cache_filename == "/path/to/image/foo.tif"

        # URL and filename without absolute path
        comp = StaticImageCompositor("name", url=remote_tif, filename="bar.tif")
        assert comp._url == remote_tif
        assert comp._cache_filename == "bar.tif"

        # No URL, filename without absolute path, use default data_dir from config
        with mock.patch("os.path.exists") as exists:
            exists.return_value = True
            comp = StaticImageCompositor("name", filename="foo.tif")
            assert comp._url is None
            assert comp._cache_filename == os.path.join(os.path.sep, "path", "to", "image", "foo.tif")
