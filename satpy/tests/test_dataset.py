#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2019 Satpy developers
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
"""Test objects and functions in the dataset module."""

import unittest
from datetime import datetime

import numpy as np
import pytest

from satpy.dataset.dataid import DataQuery, DataID, WavelengthRange, ModifierTuple, minimal_default_keys_config
from satpy.tests.utils import make_cid, make_dataid, make_dsq


class TestDataID(unittest.TestCase):
    """Test DataID object creation and other methods."""

    def test_basic_init(self):
        """Test basic ways of creating a DataID."""
        from satpy.dataset.dataid import (
            DataID,
            default_id_keys_config as dikc,
            minimal_default_keys_config as mdkc)

        did = DataID(dikc, name="a")
        assert did['name'] == 'a'
        assert did['modifiers'] == tuple()
        DataID(dikc, name="a", wavelength=0.86)
        DataID(dikc, name="a", resolution=1000)
        DataID(dikc, name="a", calibration='radiance')
        DataID(dikc, name="a", wavelength=0.86, resolution=250,
               calibration='radiance')
        DataID(dikc, name="a", wavelength=0.86, resolution=250,
               calibration='radiance', modifiers=('sunz_corrected',))
        with pytest.raises(ValueError):
            DataID(dikc, wavelength=0.86)
        did = DataID(mdkc, name='comp24', resolution=500)
        assert did['resolution'] == 500

    def test_init_bad_modifiers(self):
        """Test that modifiers are a tuple."""
        from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
        self.assertRaises(TypeError, DataID, dikc, name="a", modifiers="str")

    def test_compare_no_wl(self):
        """Compare fully qualified wavelength ID to no wavelength ID."""
        from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
        d1 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3))
        d2 = DataID(dikc, name="a", wavelength=None)

        # this happens when sorting IDs during dependency checks
        self.assertFalse(d1 < d2)
        self.assertTrue(d2 < d1)

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
        with pytest.raises(ValueError):
            DataID(dikc, name='C05', calibration='_bad_')

    def test_is_modified(self):
        """Test that modifications are detected properly."""
        from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
        d1 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=('hej',))
        d2 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=tuple())

        assert d1.is_modified()
        assert not d2.is_modified()

    def test_create_less_modified_query(self):
        """Test that modifications are popped correctly."""
        from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
        d1 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=('hej',))
        d2 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=tuple())

        assert not d1.create_less_modified_query()['modifiers']
        assert not d2.create_less_modified_query()['modifiers']


class TestCombineMetadata(unittest.TestCase):
    """Test how metadata is combined."""

    def setUp(self):
        """Set up the test case."""
        self.datetime_dts = (
            {'start_time': datetime(2018, 2, 1, 11, 58, 0)},
            {'start_time': datetime(2018, 2, 1, 11, 59, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 0, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 1, 0)},
            {'start_time': datetime(2018, 2, 1, 12, 2, 0)},
        )

    def test_average_datetimes(self):
        """Test the average_datetimes helper function."""
        from satpy.dataset.metadata import average_datetimes
        dts = (
            datetime(2018, 2, 1, 11, 58, 0),
            datetime(2018, 2, 1, 11, 59, 0),
            datetime(2018, 2, 1, 12, 0, 0),
            datetime(2018, 2, 1, 12, 1, 0),
            datetime(2018, 2, 1, 12, 2, 0),
        )
        ret = average_datetimes(dts)
        self.assertEqual(dts[2], ret)

    def test_combine_times_with_averaging(self):
        """Test the combine_metadata with times with averaging."""
        from satpy.dataset.metadata import combine_metadata
        ret = combine_metadata(*self.datetime_dts)
        self.assertEqual(self.datetime_dts[2]['start_time'], ret['start_time'])

    def test_combine_times_without_averaging(self):
        """Test the combine_metadata with times without averaging."""
        from satpy.dataset.metadata import combine_metadata
        ret = combine_metadata(*self.datetime_dts, average_times=False)
        # times are not equal so don't include it in the final result
        self.assertNotIn('start_time', ret)

    def test_combine_arrays(self):
        """Test the combine_metadata with arrays."""
        from satpy.dataset.metadata import combine_metadata
        from numpy import arange, ones
        from xarray import DataArray
        dts = [
                {"quality": (arange(25) % 2).reshape(5, 5).astype("?")},
                {"quality": (arange(1, 26) % 3).reshape(5, 5).astype("?")},
                {"quality": ones((5, 5,), "?")},
        ]
        assert "quality" not in combine_metadata(*dts)
        dts2 = [{"quality": DataArray(d["quality"])} for d in dts]
        assert "quality" not in combine_metadata(*dts2)
        # the ancillary_variables attribute is actually a list of data arrays
        dts3 = [{"quality": [d["quality"]]} for d in dts]
        assert "quality" not in combine_metadata(*dts3)
        # check cases with repeated arrays
        dts4 = [
                {"quality": dts[0]["quality"]},
                {"quality": dts[0]["quality"]},
                ]
        assert "quality" in combine_metadata(*dts4)
        dts5 = [
                {"quality": dts3[0]["quality"]},
                {"quality": dts3[0]["quality"]},
                ]
        assert "quality" in combine_metadata(*dts5)
        # check with other types
        dts6 = [
                DataArray(arange(5), attrs=dts[0]),
                DataArray(arange(5), attrs=dts[0]),
                DataArray(arange(5), attrs=dts[1]),
                object()
              ]
        assert "quality" not in combine_metadata(*dts6)

    def test_combine_identical_numpy_scalars(self):
        """Test combining idendical fill values."""
        from satpy.dataset.metadata import combine_metadata
        test_metadata = [{'_FillValue': np.uint16(42)}, {'_FillValue': np.uint16(42)}]
        assert combine_metadata(*test_metadata) == {'_FillValue': 42}

    def test_combine_empty_metadata(self):
        """Test combining empty metadata."""
        from satpy.dataset.metadata import combine_metadata
        test_metadata = [{}, {}]
        assert combine_metadata(*test_metadata) == {}

    def test_combine_nans(self):
        """Test combining nan fill values."""
        from satpy.dataset.metadata import combine_metadata
        test_metadata = [{'_FillValue': np.nan}, {'_FillValue': np.nan}]
        assert combine_metadata(*test_metadata) == {'_FillValue': np.nan}

    def test_combine_numpy_arrays(self):
        """Test combining values that are numpy arrays."""
        from satpy.dataset.metadata import combine_metadata
        test_metadata = [{'valid_range': np.array([0., 0.00032], dtype=np.float32)},
                         {'valid_range': np.array([0., 0.00032], dtype=np.float32)},
                         {'valid_range': np.array([0., 0.00032], dtype=np.float32)}]
        result = combine_metadata(*test_metadata)
        assert np.allclose(result['valid_range'], np.array([0., 0.00032], dtype=np.float32))

    def test_combine_dask_arrays(self):
        """Test combining values that are dask arrays."""
        from satpy.dataset.metadata import combine_metadata
        import dask.array as da
        test_metadata = [{'valid_range': da.from_array(np.array([0., 0.00032], dtype=np.float32))},
                         {'valid_range': da.from_array(np.array([0., 0.00032], dtype=np.float32))}]
        result = combine_metadata(*test_metadata)
        assert 'valid_range' not in result

    def test_combine_real_world_mda(self):
        """Test with real data."""
        mda_objects = ({'_FillValue': np.nan,
                        'valid_range': np.array([0., 0.00032], dtype=np.float32),
                        'ancillary_variables': ['cpp_status_flag',
                                                'cpp_conditions',
                                                'cpp_quality',
                                                'cpp_reff_pal',
                                                '-'],
                        'platform_name': 'NOAA-20',
                        'sensor': {'viirs'}},
                       {'_FillValue': np.nan,
                        'valid_range': np.array([0., 0.00032], dtype=np.float32),
                        'ancillary_variables': ['cpp_status_flag',
                                                'cpp_conditions',
                                                'cpp_quality',
                                                'cpp_reff_pal',
                                                '-'],
                        'platform_name': 'NOAA-20',
                        'sensor': {'viirs'}})

        expected = {'_FillValue': np.nan,
                    'valid_range': np.array([0., 0.00032], dtype=np.float32),
                    'ancillary_variables': ['cpp_status_flag',
                                            'cpp_conditions',
                                            'cpp_quality',
                                            'cpp_reff_pal',
                                            '-'],
                    'platform_name': 'NOAA-20',
                    'sensor': {'viirs'}}

        from satpy.dataset.metadata import combine_metadata
        result = combine_metadata(*mda_objects)
        assert np.allclose(result.pop('_FillValue'), expected.pop('_FillValue'), equal_nan=True)
        assert np.allclose(result.pop('valid_range'), expected.pop('valid_range'))
        assert result == expected

    def test_combine_one_metadata_object(self):
        """Test combining one metadata object."""
        mda_objects = ({'_FillValue': np.nan,
                        'valid_range': np.array([0., 0.00032], dtype=np.float32),
                        'ancillary_variables': ['cpp_status_flag',
                                                'cpp_conditions',
                                                'cpp_quality',
                                                'cpp_reff_pal',
                                                '-'],
                        'platform_name': 'NOAA-20',
                        'sensor': {'viirs'}},)

        expected = {'_FillValue': np.nan,
                    'valid_range': np.array([0., 0.00032], dtype=np.float32),
                    'ancillary_variables': ['cpp_status_flag',
                                            'cpp_conditions',
                                            'cpp_quality',
                                            'cpp_reff_pal',
                                            '-'],
                    'platform_name': 'NOAA-20',
                    'sensor': {'viirs'}}

        from satpy.dataset.metadata import combine_metadata
        result = combine_metadata(*mda_objects)
        assert np.allclose(result.pop('_FillValue'), expected.pop('_FillValue'), equal_nan=True)
        assert np.allclose(result.pop('valid_range'), expected.pop('valid_range'))
        assert result == expected


def test_dataid():
    """Test the DataID object."""
    from satpy.dataset.dataid import DataID, WavelengthRange, ModifierTuple, ValueList

    # Check that enum is translated to type.
    did = make_dataid()
    assert issubclass(did._id_keys['calibration']['type'], ValueList)
    assert 'enum' not in did._id_keys['calibration']

    # Check that None is never a valid value
    did = make_dataid(name='cheese_shops', resolution=None)
    assert 'resolution' not in did
    assert 'None' not in did.__repr__()
    with pytest.raises(ValueError):
        make_dataid(name=None, resolution=1000)

    # Check that defaults are applied correctly
    assert did['modifiers'] == ModifierTuple()

    # Check that from_dict creates a distinct instance...
    did2 = did.from_dict(dict(name='cheese_shops', resolution=None))
    assert did is not did2
    # ...But is equal
    assert did2 == did

    # Check that the instance is immutable
    with pytest.raises(TypeError):
        did['resolution'] = 1000

    # Check that a missing required field crashes
    with pytest.raises(ValueError):
        make_dataid(resolution=1000)

    # Check to_dict
    assert did.to_dict() == dict(name='cheese_shops', modifiers=tuple())

    # Check repr
    did = make_dataid(name='VIS008', resolution=111)
    assert repr(did) == "DataID(name='VIS008', resolution=111, modifiers=())"

    # Check inequality
    default_id_keys_config = {'name': None,
                              'wavelength': {
                                'type': WavelengthRange,
                              },
                              'resolution': None,
                              'calibration': {
                                'enum': [
                                    'reflectance',
                                    'brightness_temperature',
                                    'radiance',
                                    'counts'
                                    ]
                              },
                              'modifiers': {
                                'default': ModifierTuple(),
                                'type': ModifierTuple,
                              },
                              }
    assert DataID(default_id_keys_config, wavelength=10) != DataID(default_id_keys_config, name="VIS006")


def test_dataid_equal_if_enums_different():
    """Check that dataids with different enums but same items are equal."""
    from satpy.dataset.dataid import DataID, WavelengthRange, ModifierTuple
    id_keys_config1 = {'name': None,
                       'wavelength': {
                           'type': WavelengthRange,
                       },
                       'resolution': None,
                       'calibration': {
                           'enum': [
                               'c1',
                               'c2',
                               'c3',
                           ]
                       },
                       'modifiers': {
                           'default': ModifierTuple(),
                           'type': ModifierTuple,
                       },
                       }

    id_keys_config2 = {'name': None,
                       'wavelength': {
                           'type': WavelengthRange,
                       },
                       'resolution': None,
                       'calibration': {
                           'enum': [
                               'c1',
                               'c1.5',
                               'c2',
                               'c2.5',
                               'c3'
                           ]
                       },
                       'modifiers': {
                           'default': ModifierTuple(),
                           'type': ModifierTuple,
                       },
                       }
    assert DataID(id_keys_config1, name='ni', calibration='c2') == DataID(id_keys_config2, name="ni", calibration='c2')


def test_dataid_copy():
    """Test copying a DataID."""
    from satpy.dataset.dataid import DataID, default_id_keys_config as dikc
    from copy import deepcopy

    did = DataID(dikc, name="a", resolution=1000)
    did2 = deepcopy(did)
    assert did2 == did
    assert did2.id_keys == did.id_keys


def test_dataid_pickle():
    """Test dataid pickling roundtrip."""
    from satpy.tests.utils import make_dataid
    import pickle
    did = make_dataid(name='hi', wavelength=(10, 11, 12), resolution=1000, calibration='radiance')
    assert did == pickle.loads(pickle.dumps(did))


class TestDataQuery:
    """Test case for data queries."""

    def test_dataquery(self):
        """Test DataQuery objects."""
        from satpy.dataset import DataQuery

        DataQuery(name='cheese_shops')

        # Check repr
        did = DataQuery(name='VIS008', resolution=111)
        assert repr(did) == "DataQuery(name='VIS008', resolution=111)"

        # Check inequality
        assert DataQuery(wavelength=10) != DataQuery(name="VIS006")

    def test_is_modified(self):
        """Test that modifications are detected properly."""
        from satpy.dataset import DataQuery
        d1 = DataQuery(name="a", wavelength=0.2, modifiers=('hej',))
        d2 = DataQuery(name="a", wavelength=0.2, modifiers=tuple())

        assert d1.is_modified()
        assert not d2.is_modified()

    def test_create_less_modified_query(self):
        """Test that modifications are popped correctly."""
        from satpy.dataset import DataQuery
        d1 = DataQuery(name="a", wavelength=0.2, modifiers=('hej',))
        d2 = DataQuery(name="a", wavelength=0.2, modifiers=tuple())

        assert not d1.create_less_modified_query()['modifiers']
        assert not d2.create_less_modified_query()['modifiers']


class TestIDQueryInteractions(unittest.TestCase):
    """Test the interactions between DataIDs and DataQuerys."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.default_id_keys_config = {
            'name': {
                'required': True,
            },
            'wavelength': {
                'type': WavelengthRange,
            },
            'resolution': None,
            'calibration': {
                'enum': [
                    'reflectance',
                    'brightness_temperature',
                    'radiance',
                    'counts'
                ]
            },
            'modifiers': {
                'default': ModifierTuple(),
                'type': ModifierTuple,
            },
        }

    def test_hash_equality(self):
        """Test hash equality."""
        dq = DataQuery(modifiers=tuple(), name='cheese_shops')
        did = DataID(self.default_id_keys_config, name='cheese_shops')
        assert hash(dq) == hash(did)

    def test_id_filtering(self):
        """Check did filtering."""
        dq = DataQuery(modifiers=tuple(), name='cheese_shops')
        did = DataID(self.default_id_keys_config, name='cheese_shops')
        did2 = DataID(self.default_id_keys_config, name='ni')
        res = dq.filter_dataids([did2, did])
        assert len(res) == 1
        assert res[0] == did

        dataid_container = [DataID(self.default_id_keys_config,
                                   name='ds1',
                                   resolution=250,
                                   calibration='reflectance',
                                   modifiers=tuple())]
        dq = DataQuery(wavelength=0.22, modifiers=tuple())
        assert len(dq.filter_dataids(dataid_container)) == 0
        dataid_container = [DataID(minimal_default_keys_config,
                                   name='natural_color')]
        dq = DataQuery(name='natural_color', resolution=250)
        assert len(dq.filter_dataids(dataid_container)) == 1

        dq = make_dsq(wavelength=0.22, modifiers=('mod1',))
        did = make_cid(name='static_image')
        assert len(dq.filter_dataids([did])) == 0

    def test_inequality(self):
        """Check (in)equality."""
        assert DataQuery(wavelength=10) != DataID(self.default_id_keys_config, name="VIS006")

    def test_sort_dataids(self):
        """Check dataid sorting."""
        dq = DataQuery(name='cheese_shops', wavelength=2, modifiers='*')
        did = DataID(self.default_id_keys_config, name='cheese_shops', wavelength=(1, 2, 3))
        did2 = DataID(self.default_id_keys_config, name='cheese_shops', wavelength=(1.1, 2.1, 3.1))
        dsids, distances = dq.sort_dataids([did2, did])
        assert list(dsids) == [did, did2]
        assert np.allclose(distances, [0, 0.1])

        dq = DataQuery(name='cheese_shops')
        did = DataID(self.default_id_keys_config, name='cheese_shops', resolution=200)
        did2 = DataID(self.default_id_keys_config, name='cheese_shops', resolution=400)
        dsids, distances = dq.sort_dataids([did2, did])
        assert list(dsids) == [did, did2]
        assert distances[0] < distances[1]

        did = DataID(self.default_id_keys_config, name='cheese_shops', calibration='counts')
        did2 = DataID(self.default_id_keys_config, name='cheese_shops', calibration='reflectance')
        dsids, distances = dq.sort_dataids([did2, did])
        assert list(dsids) == [did2, did]
        assert distances[0] < distances[1]

        did = DataID(self.default_id_keys_config, name='cheese_shops', modifiers=tuple())
        did2 = DataID(self.default_id_keys_config, name='cheese_shops', modifiers=tuple(['out_of_stock']))
        dsids, distances = dq.sort_dataids([did2, did])
        assert list(dsids) == [did, did2]
        assert distances[0] < distances[1]

    def test_sort_dataids_with_different_set_of_keys(self):
        """Check sorting data ids when the query has a different set of keys."""
        dq = DataQuery(name='solar_zenith_angle', calibration='reflectance')
        dids = [DataID(self.default_id_keys_config, name='solar_zenith_angle', resolution=1000, modifiers=()),
                DataID(self.default_id_keys_config, name='solar_zenith_angle', resolution=500, modifiers=()),
                DataID(self.default_id_keys_config, name='solar_zenith_angle', resolution=250, modifiers=())]
        dsids, distances = dq.sort_dataids(dids)
        assert distances[0] < distances[1]
        assert distances[1] < distances[2]


def test_wavelength_range():
    """Test the wavelength range object."""
    from satpy.dataset.dataid import WavelengthRange

    wr = WavelengthRange(1, 2, 3)
    assert 1.2 == wr
    assert .9 != wr
    assert wr == (1, 2, 3)
    assert wr == (1, 2, 3, 'µm')

    # Check containement
    assert 1.2 in wr
    assert .9 not in wr
    assert WavelengthRange(1, 2, 3) in wr
    assert WavelengthRange(1.1, 2.2, 3.3) not in wr
    assert WavelengthRange(1.2, 2, 2.8) in wr
    assert WavelengthRange(10, 20, 30) not in wr
    assert 'bla' not in wr
    assert None not in wr
    wr2 = WavelengthRange(1, 2, 3, 'µm')
    assert wr2 in wr
    wr2 = WavelengthRange(1, 2, 3, 'nm')
    with pytest.raises(NotImplementedError):
        wr2 in wr  # noqa

    # Check __str__
    assert str(wr) == "2 µm (1-3 µm)"
    assert str(wr2) == "2 nm (1-3 nm)"


def test_wavelength_range_cf_roundtrip():
    """Test the wavelength range object roundtrip to cf."""
    from satpy.dataset.dataid import WavelengthRange

    wr = WavelengthRange(1, 2, 3)

    assert WavelengthRange.from_cf(wr.to_cf()) == wr
    assert WavelengthRange.from_cf([str(item) for item in wr]) == wr
