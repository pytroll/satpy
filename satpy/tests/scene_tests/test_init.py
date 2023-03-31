# Copyright (c) 2010-2023 Satpy developers
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
"""Unit tests for Scene creation."""

import os
import random
import string
from copy import deepcopy
from unittest import mock

import pytest

import satpy
from satpy import Scene
from satpy.tests.utils import FAKE_FILEHANDLER_END, FAKE_FILEHANDLER_START, spy_decorator

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - include_test_etc


@pytest.mark.usefixtures("include_test_etc")
class TestScene:
    """Test the scene class."""

    def test_init(self):
        """Test scene initialization."""
        with mock.patch('satpy.scene.Scene._create_reader_instances') as cri:
            cri.return_value = {}
            Scene(filenames=['bla'], reader='blo')
            cri.assert_called_once_with(filenames=['bla'], reader='blo',
                                        reader_kwargs=None)

    def test_init_str_filename(self):
        """Test initializing with a single string as filenames."""
        pytest.raises(ValueError, Scene, reader='blo', filenames='test.nc')

    def test_start_end_times(self):
        """Test start and end times for a scene."""
        scene = Scene(filenames=['fake1_1.txt'], reader='fake1')
        assert scene.start_time == FAKE_FILEHANDLER_START
        assert scene.end_time == FAKE_FILEHANDLER_END

    def test_init_preserve_reader_kwargs(self):
        """Test that the initialization preserves the kwargs."""
        cri = spy_decorator(Scene._create_reader_instances)
        with mock.patch('satpy.scene.Scene._create_reader_instances', cri):
            reader_kwargs = {'calibration_type': 'gsics'}
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          filter_parameters={'area': 'euron1'},
                          reader_kwargs=reader_kwargs)
            assert reader_kwargs is not cri.mock.call_args[1]['reader_kwargs']
            assert scene.start_time == FAKE_FILEHANDLER_START
            assert scene.end_time == FAKE_FILEHANDLER_END

    def test_init_alone(self):
        """Test simple initialization."""
        scn = Scene()
        assert not scn._readers, 'Empty scene should not load any readers'

    def test_init_no_files(self):
        """Test that providing an empty list of filenames fails."""
        pytest.raises(ValueError, Scene, reader='viirs_sdr', filenames=[])

    def test_create_reader_instances_with_filenames(self):
        """Test creating a reader providing filenames."""
        filenames = ["bla", "foo", "bar"]
        reader_name = None
        with mock.patch('satpy.scene.load_readers') as findermock:
            Scene(filenames=filenames)
            findermock.assert_called_once_with(
                filenames=filenames,
                reader=reader_name,
                reader_kwargs=None,
            )

    def test_init_with_empty_filenames(self):
        """Test initialization with empty filename list."""
        filenames = []
        Scene(filenames=filenames)

    def test_init_with_fsfile(self):
        """Test initialisation with FSFile objects."""
        from satpy.readers import FSFile

        # We should not mock _create_reader_instances here, because in
        # https://github.com/pytroll/satpy/issues/1605 satpy fails with
        # TypeError within that method if passed an FSFile instance.
        # Instead rely on the ValueError that satpy raises if no readers
        # are found.
        # Choose random filename that doesn't exist.  Not using tempfile here,
        # because tempfile creates files and we don't want that here.
        fsf = FSFile("".join(random.choices(string.printable, k=50)))
        with pytest.raises(ValueError, match="No supported files found"):
            Scene(filenames=[fsf], reader=[])

    def test_create_reader_instances_with_reader(self):
        """Test createring a reader instance providing the reader name."""
        reader = "foo"
        filenames = ["1", "2", "3"]
        with mock.patch('satpy.scene.load_readers') as findermock:
            findermock.return_value = {}
            Scene(reader=reader, filenames=filenames)
            findermock.assert_called_once_with(reader=reader,
                                               filenames=filenames,
                                               reader_kwargs=None,
                                               )

    def test_create_reader_instances_with_reader_kwargs(self):
        """Test creating a reader instance with reader kwargs."""
        from satpy.readers.yaml_reader import FileYAMLReader
        reader_kwargs = {'calibration_type': 'gsics'}
        filter_parameters = {'area': 'euron1'}
        reader_kwargs2 = {'calibration_type': 'gsics', 'filter_parameters': filter_parameters}

        rinit = spy_decorator(FileYAMLReader.create_filehandlers)
        with mock.patch('satpy.readers.yaml_reader.FileYAMLReader.create_filehandlers', rinit):
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          filter_parameters={'area': 'euron1'},
                          reader_kwargs=reader_kwargs)
            del scene
            assert reader_kwargs == rinit.mock.call_args[1]['fh_kwargs']
            rinit.mock.reset_mock()
            scene = Scene(filenames=['fake1_1.txt'],
                          reader='fake1',
                          reader_kwargs=reader_kwargs2)
            assert reader_kwargs == rinit.mock.call_args[1]['fh_kwargs']
            del scene

    def test_create_multiple_reader_different_kwargs(self, include_test_etc):
        """Test passing different kwargs to different readers."""
        from satpy.readers import load_reader
        with mock.patch.object(satpy.readers, 'load_reader', wraps=load_reader) as lr:
            Scene(filenames={"fake1_1ds": ["fake1_1ds_1.txt"],
                             "fake2_1ds": ["fake2_1ds_1.txt"]},
                  reader_kwargs={
                      "fake1_1ds": {"mouth": "omegna"},
                      "fake2_1ds": {"mouth": "varallo"}
                  })
            lr.assert_has_calls([
                mock.call([os.path.join(include_test_etc, 'readers', 'fake1_1ds.yaml')], mouth="omegna"),
                mock.call([os.path.join(include_test_etc, 'readers', 'fake2_1ds.yaml')], mouth="varallo")])

    def test_storage_options_from_reader_kwargs_no_options(self):
        """Test getting storage options from reader kwargs.

        Case where there are no options given.
        """
        filenames = ["s3://data-bucket/file1", "s3://data-bucket/file2", "s3://data-bucket/file3"]
        with mock.patch('satpy.scene.load_readers'):
            with mock.patch('fsspec.open_files') as open_files:
                Scene(filenames=filenames)
                open_files.assert_called_once_with(filenames)

    def test_storage_options_from_reader_kwargs_single_dict_no_options(self):
        """Test getting storage options from reader kwargs for remote files.

        Case where a single dict is given for all readers without storage options.
        """
        filenames = ["s3://data-bucket/file1", "s3://data-bucket/file2", "s3://data-bucket/file3"]
        reader_kwargs = {'reader_opt': 'foo'}
        with mock.patch('satpy.scene.load_readers'):
            with mock.patch('fsspec.open_files') as open_files:
                Scene(filenames=filenames, reader_kwargs=reader_kwargs)
                open_files.assert_called_once_with(filenames)

    @pytest.mark.parametrize("reader_kwargs", [{}, {'reader_opt': 'foo'}])
    def test_storage_options_from_reader_kwargs_single_dict(self, reader_kwargs):
        """Test getting storage options from reader kwargs.

        Case where a single dict is given for all readers with some common storage options.
        """
        filenames = ["s3://data-bucket/file1", "s3://data-bucket/file2", "s3://data-bucket/file3"]
        expected_reader_kwargs = reader_kwargs.copy()
        storage_options = {'option1': '1'}
        reader_kwargs['storage_options'] = storage_options
        orig_reader_kwargs = deepcopy(reader_kwargs)
        with mock.patch('satpy.scene.load_readers') as load_readers:
            with mock.patch('fsspec.open_files') as open_files:
                Scene(filenames=filenames, reader_kwargs=reader_kwargs)
                call_ = load_readers.mock_calls[0]
                assert call_.kwargs['reader_kwargs'] == expected_reader_kwargs
                open_files.assert_called_once_with(filenames, **storage_options)
                assert reader_kwargs == orig_reader_kwargs

    def test_storage_options_from_reader_kwargs_per_reader(self):
        """Test getting storage options from reader kwargs.

        Case where each reader have their own storage options.
        """
        filenames = {
            "reader1": ["s3://data-bucket/file1"],
            "reader2": ["s3://data-bucket/file2"],
            "reader3": ["s3://data-bucket/file3"],
        }
        storage_options_1 = {'option1': '1'}
        storage_options_2 = {'option2': '2'}
        storage_options_3 = {'option3': '3'}
        reader_kwargs = {
            "reader1": {'reader_opt_1': 'foo'},
            "reader2": {'reader_opt_2': 'bar'},
            "reader3": {'reader_opt_3': 'baz'},
        }
        expected_reader_kwargs = deepcopy(reader_kwargs)
        reader_kwargs['reader1']['storage_options'] = storage_options_1
        reader_kwargs['reader2']['storage_options'] = storage_options_2
        reader_kwargs['reader3']['storage_options'] = storage_options_3
        orig_reader_kwargs = deepcopy(reader_kwargs)

        with mock.patch('satpy.scene.load_readers') as load_readers:
            with mock.patch('fsspec.open_files') as open_files:
                Scene(filenames=filenames, reader_kwargs=reader_kwargs)
                call_ = load_readers.mock_calls[0]
                assert call_.kwargs['reader_kwargs'] == expected_reader_kwargs
                assert mock.call(filenames["reader1"], **storage_options_1) in open_files.mock_calls
                assert mock.call(filenames["reader2"], **storage_options_2) in open_files.mock_calls
                assert mock.call(filenames["reader3"], **storage_options_3) in open_files.mock_calls
                assert reader_kwargs == orig_reader_kwargs

    def test_storage_options_from_reader_kwargs_per_reader_and_global(self):
        """Test getting storage options from reader kwargs.

        Case where each reader have their own storage options and there are
        global options to merge.
        """
        filenames = {
            "reader1": ["s3://data-bucket/file1"],
            "reader2": ["s3://data-bucket/file2"],
            "reader3": ["s3://data-bucket/file3"],
        }
        reader_kwargs = {
            "reader1": {'reader_opt_1': 'foo', 'storage_options': {'option1': '1'}},
            "reader2": {'reader_opt_2': 'bar', 'storage_options': {'option2': '2'}},
            "storage_options": {"endpoint_url": "url"},
        }
        orig_reader_kwargs = deepcopy(reader_kwargs)

        with mock.patch('satpy.scene.load_readers'):
            with mock.patch('fsspec.open_files') as open_files:
                Scene(filenames=filenames, reader_kwargs=reader_kwargs)
                assert mock.call(filenames["reader1"], option1='1', endpoint_url='url') in open_files.mock_calls
                assert mock.call(filenames["reader2"], option2='2', endpoint_url='url') in open_files.mock_calls
                assert reader_kwargs == orig_reader_kwargs
