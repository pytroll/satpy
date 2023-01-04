#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2015-2022 Satpy developers
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

from satpy.dataset.dataid import DataID, DataQuery, ModifierTuple, WavelengthRange, minimal_default_keys_config
from satpy.readers.pmw_channels_definitions import FrequencyDoubleSideBand, FrequencyQuadrupleSideBand, FrequencyRange
from satpy.tests.utils import make_cid, make_dataid, make_dsq


class TestDataID(unittest.TestCase):
    """Test DataID object creation and other methods."""

    def test_basic_init(self):
        """Test basic ways of creating a DataID."""
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
        from satpy.dataset.dataid import minimal_default_keys_config as mdkc

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
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
        self.assertRaises(TypeError, DataID, dikc, name="a", modifiers="str")

    def test_compare_no_wl(self):
        """Compare fully qualified wavelength ID to no wavelength ID."""
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
        d1 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3))
        d2 = DataID(dikc, name="a", wavelength=None)

        # this happens when sorting IDs during dependency checks
        self.assertFalse(d1 < d2)
        self.assertTrue(d2 < d1)

    def test_bad_calibration(self):
        """Test that asking for a bad calibration fails."""
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
        with pytest.raises(ValueError):
            DataID(dikc, name='C05', calibration='_bad_')

    def test_is_modified(self):
        """Test that modifications are detected properly."""
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
        d1 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=('hej',))
        d2 = DataID(dikc, name="a", wavelength=(0.1, 0.2, 0.3), modifiers=tuple())

        assert d1.is_modified()
        assert not d2.is_modified()

    def test_create_less_modified_query(self):
        """Test that modifications are popped correctly."""
        from satpy.dataset.dataid import DataID
        from satpy.dataset.dataid import default_id_keys_config as dikc
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
        from numpy import arange, ones
        from xarray import DataArray

        from satpy.dataset.metadata import combine_metadata
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

    def test_combine_lists_identical(self):
        """Test combine metadata with identical lists."""
        from satpy.dataset.metadata import combine_metadata
        metadatas = [
            {'prerequisites': [1, 2, 3, 4]},
            {'prerequisites': [1, 2, 3, 4]},
        ]
        res = combine_metadata(*metadatas)
        assert res['prerequisites'] == [1, 2, 3, 4]

    def test_combine_lists_same_size_diff_values(self):
        """Test combine metadata with lists with different values."""
        from satpy.dataset.metadata import combine_metadata
        metadatas = [
            {'prerequisites': [1, 2, 3, 4]},
            {'prerequisites': [1, 2, 3, 5]},
        ]
        res = combine_metadata(*metadatas)
        assert 'prerequisites' not in res

    def test_combine_lists_different_size(self):
        """Test combine metadata with different size lists."""
        from satpy.dataset.metadata import combine_metadata
        metadatas = [
            {'prerequisites': [1, 2, 3, 4]},
            {'prerequisites': []},
        ]
        res = combine_metadata(*metadatas)
        assert 'prerequisites' not in res

        metadatas = [
            {'prerequisites': [1, 2, 3, 4]},
            {'prerequisites': [1, 2, 3]},
        ]
        res = combine_metadata(*metadatas)
        assert 'prerequisites' not in res

    def test_combine_identical_numpy_scalars(self):
        """Test combining identical fill values."""
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
        import dask.array as da

        from satpy.dataset.metadata import combine_metadata
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
                        'sensor': {'viirs'},
                        'raw_metadata': {'foo': {'bar': np.array([1, 2, 3])}}},
                       {'_FillValue': np.nan,
                        'valid_range': np.array([0., 0.00032], dtype=np.float32),
                        'ancillary_variables': ['cpp_status_flag',
                                                'cpp_conditions',
                                                'cpp_quality',
                                                'cpp_reff_pal',
                                                '-'],
                        'platform_name': 'NOAA-20',
                        'sensor': {'viirs'},
                        'raw_metadata': {'foo': {'bar': np.array([1, 2, 3])}}})

        expected = {'_FillValue': np.nan,
                    'valid_range': np.array([0., 0.00032], dtype=np.float32),
                    'ancillary_variables': ['cpp_status_flag',
                                            'cpp_conditions',
                                            'cpp_quality',
                                            'cpp_reff_pal',
                                            '-'],
                    'platform_name': 'NOAA-20',
                    'sensor': {'viirs'},
                    'raw_metadata': {'foo': {'bar': np.array([1, 2, 3])}}}

        from satpy.dataset.metadata import combine_metadata
        result = combine_metadata(*mda_objects)
        assert np.allclose(result.pop('_FillValue'), expected.pop('_FillValue'), equal_nan=True)
        assert np.allclose(result.pop('valid_range'), expected.pop('valid_range'))
        np.testing.assert_equal(result.pop('raw_metadata'),
                                expected.pop('raw_metadata'))
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


def test_combine_dicts_close():
    """Test combination of dictionaries whose values are close."""
    from satpy.dataset.metadata import combine_metadata
    attrs = {
        'raw_metadata': {
            'a': 1,
            'b': 'foo',
            'c': [1, 2, 3],
            'd': {
                'e': np.str_('bar'),
                'f': datetime(2020, 1, 1, 12, 15, 30),
                'g': np.array([1, 2, 3]),
            },
            'h': np.array([datetime(2020, 1, 1), datetime(2020, 1, 1)])
        }
    }
    attrs_close = {
        'raw_metadata': {
            'a': 1 + 1E-12,
            'b': 'foo',
            'c': np.array([1, 2, 3]) + 1E-12,
            'd': {
                'e': np.str_('bar'),
                'f': datetime(2020, 1, 1, 12, 15, 30),
                'g': np.array([1, 2, 3]) + 1E-12
            },
            'h': np.array([datetime(2020, 1, 1), datetime(2020, 1, 1)])
        }
    }
    test_metadata = [attrs, attrs_close]
    result = combine_metadata(*test_metadata)
    assert result == attrs


@pytest.mark.parametrize(
    "test_mda",
    [
        # a/b/c/d different
        {'a': np.array([1, 2, 3]), 'd': 123},
        {'a': {'b': np.array([4, 5, 6]), 'c': 1.0}, 'd': 'foo'},
        {'a': {'b': np.array([1, 2, 3]), 'c': 2.0}, 'd': 'foo'},
        {'a': {'b': np.array([1, 2, 3]), 'c': 1.0}, 'd': 'bar'},
        # a/b/c/d type different
        np.array([1, 2, 3]),
        {'a': {'b': 'baz', 'c': 1.0}, 'd': 'foo'},
        {'a': {'b': np.array([1, 2, 3]), 'c': 'baz'}, 'd': 'foo'},
        {'a': {'b': np.array([1, 2, 3]), 'c': 1.0}, 'd': 1.0}
    ]
)
def test_combine_dicts_different(test_mda):
    """Test combination of dictionaries differing in various ways."""
    from satpy.dataset.metadata import combine_metadata
    mda = {'a': {'b': np.array([1, 2, 3]), 'c': 1.0}, 'd': 'foo'}
    test_metadata = [{'raw_metadata': mda}, {'raw_metadata': test_mda}]
    result = combine_metadata(*test_metadata)
    assert not result


def test_dataid():
    """Test the DataID object."""
    from satpy.dataset.dataid import DataID, ModifierTuple, ValueList, WavelengthRange

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
    from satpy.dataset.dataid import DataID, ModifierTuple, WavelengthRange
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
    from copy import deepcopy

    from satpy.dataset.dataid import DataID
    from satpy.dataset.dataid import default_id_keys_config as dikc

    did = DataID(dikc, name="a", resolution=1000)
    did2 = deepcopy(did)
    assert did2 == did
    assert did2.id_keys == did.id_keys


def test_dataid_pickle():
    """Test dataid pickling roundtrip."""
    import pickle

    from satpy.tests.utils import make_dataid
    did = make_dataid(name='hi', wavelength=(10, 11, 12), resolution=1000, calibration='radiance')
    assert did == pickle.loads(pickle.dumps(did))


def test_dataid_elements_picklable():
    """Test individual elements of DataID can be pickled.

    In some cases, like in the base reader classes, the elements of a DataID
    are extracted and stored in a separate dictionary. This means that the
    internal/fancy pickle handling of DataID does not play a part.

    """
    import pickle

    from satpy.tests.utils import make_dataid
    did = make_dataid(name='hi', wavelength=(10, 11, 12), resolution=1000, calibration='radiance')
    for value in did.values():
        pickled_value = pickle.loads(pickle.dumps(value))
        assert value == pickled_value


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

    def test_seviri_hrv_has_priority_over_vis008(self):
        """Check that the HRV channel has priority over VIS008 when querying 0.8µm."""
        dids = [DataID(self.default_id_keys_config, name='HRV',
                       wavelength=WavelengthRange(min=0.5, central=0.7, max=0.9, unit='µm'), resolution=1000.134348869,
                       calibration="reflectance", modifiers=()),
                DataID(self.default_id_keys_config, name='HRV',
                       wavelength=WavelengthRange(min=0.5, central=0.7, max=0.9, unit='µm'), resolution=1000.134348869,
                       calibration="radiance", modifiers=()),
                DataID(self.default_id_keys_config, name='HRV',
                       wavelength=WavelengthRange(min=0.5, central=0.7, max=0.9, unit='µm'), resolution=1000.134348869,
                       calibration="counts", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS006',
                       wavelength=WavelengthRange(min=0.56, central=0.635, max=0.71, unit='µm'),
                       resolution=3000.403165817, calibration="reflectance", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS006',
                       wavelength=WavelengthRange(min=0.56, central=0.635, max=0.71, unit='µm'),
                       resolution=3000.403165817, calibration="radiance", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS006',
                       wavelength=WavelengthRange(min=0.56, central=0.635, max=0.71, unit='µm'),
                       resolution=3000.403165817, calibration="counts", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS008',
                       wavelength=WavelengthRange(min=0.74, central=0.81, max=0.88, unit='µm'),
                       resolution=3000.403165817, calibration="reflectance", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS008',
                       wavelength=WavelengthRange(min=0.74, central=0.81, max=0.88, unit='µm'),
                       resolution=3000.403165817, calibration="radiance", modifiers=()),
                DataID(self.default_id_keys_config, name='VIS008',
                       wavelength=WavelengthRange(min=0.74, central=0.81, max=0.88, unit='µm'),
                       resolution=3000.403165817, calibration="counts", modifiers=())]
        dq = DataQuery(wavelength=0.8)
        res, distances = dq.sort_dataids(dids)
        assert res[0].name == "HRV"


def test_frequency_quadruple_side_band_class_method_convert():
    """Test the frequency double side band object: test the class method convert."""
    frq_qdsb = FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036)

    res = frq_qdsb.convert(57.37)
    assert res == 57.37

    res = frq_qdsb.convert({'central': 57.0, 'side': 0.322, 'sideside': 0.05, 'bandwidth': 0.036})
    assert res == FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036)


def test_frequency_quadruple_side_band_channel_str():
    """Test the frequency quadruple side band object: test the band description."""
    frq_qdsb1 = FrequencyQuadrupleSideBand(57.0, 0.322, 0.05, 0.036)
    frq_qdsb2 = FrequencyQuadrupleSideBand(57000, 322, 50, 36, 'MHz')

    assert str(frq_qdsb1) == "central=57.0 GHz ±0.322 ±0.05 width=0.036 GHz"
    assert str(frq_qdsb2) == "central=57000 MHz ±322 ±50 width=36 MHz"


def test_frequency_quadruple_side_band_channel_equality():
    """Test the frequency quadruple side band object: check if two bands are 'equal'."""
    frq_qdsb = FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036)
    assert frq_qdsb is not None
    assert frq_qdsb < FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.04)
    assert frq_qdsb < FrequencyQuadrupleSideBand(58, 0.322, 0.05, 0.036)
    assert frq_qdsb < ((58, 0.322, 0.05, 0.036))
    assert frq_qdsb > FrequencyQuadrupleSideBand(57, 0.322, 0.04, 0.01)
    assert frq_qdsb > None
    assert (frq_qdsb < None) is False

    assert 57 != frq_qdsb
    assert 57.372 == frq_qdsb
    assert 56.646 == frq_qdsb
    assert 56.71 == frq_qdsb

    assert frq_qdsb != FrequencyQuadrupleSideBand(57, 0.322, 0.1, 0.040)

    frq_qdsb = None
    assert FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036) != frq_qdsb
    assert frq_qdsb < FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.04)


def test_frequency_quadruple_side_band_channel_distances():
    """Test the frequency quadruple side band object: get the distance between two bands."""
    frq_qdsb = FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036)
    mydist = frq_qdsb.distance([57, 0.322, 0.05, 0.036])

    frq_dict = {'central': 57, 'side': 0.322, 'sideside': 0.05,
                'bandwidth': 0.036, 'unit': 'GHz'}
    mydist = frq_qdsb.distance(frq_dict)
    assert mydist == np.inf

    mydist = frq_qdsb.distance(57.372)
    assert mydist == 0.0

    mydist = frq_qdsb.distance(FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036))
    assert mydist == 0.0

    mydist = frq_qdsb.distance(57.38)
    np.testing.assert_almost_equal(mydist, 0.008)

    mydist = frq_qdsb.distance(57)
    assert mydist == np.inf

    mydist = frq_qdsb.distance((57, 0.322, 0.05, 0.018))
    assert mydist == np.inf


def test_frequency_quadruple_side_band_channel_containment():
    """Test the frequency quadruple side band object: check if one band contains another."""
    frq_qdsb = FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.036)

    assert 57 not in frq_qdsb
    assert 57.373 in frq_qdsb

    with pytest.raises(NotImplementedError):
        assert frq_qdsb in FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.05)

    frq_qdsb = None
    assert (frq_qdsb in FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.05)) is False

    assert '57' not in FrequencyQuadrupleSideBand(57, 0.322, 0.05, 0.05)


def test_frequency_double_side_band_class_method_convert():
    """Test the frequency double side band object: test the class method convert."""
    frq_dsb = FrequencyDoubleSideBand(183, 7, 2)

    res = frq_dsb.convert(185)
    assert res == 185

    res = frq_dsb.convert({'central': 185, 'side': 7, 'bandwidth': 2})
    assert res == FrequencyDoubleSideBand(185, 7, 2)


def test_frequency_double_side_band_channel_str():
    """Test the frequency double side band object: test the band description."""
    frq_dsb1 = FrequencyDoubleSideBand(183, 7, 2)
    frq_dsb2 = FrequencyDoubleSideBand(183000, 7000, 2000, 'MHz')

    assert str(frq_dsb1) == "central=183 GHz ±7 width=2 GHz"
    assert str(frq_dsb2) == "central=183000 MHz ±7000 width=2000 MHz"


def test_frequency_double_side_band_channel_equality():
    """Test the frequency double side band object: check if two bands are 'equal'."""
    frq_dsb = FrequencyDoubleSideBand(183, 7, 2)
    assert frq_dsb is not None
    assert 183 != frq_dsb
    assert 190 == frq_dsb
    assert 176 == frq_dsb
    assert 175.5 == frq_dsb

    assert frq_dsb != FrequencyDoubleSideBand(183, 6.5, 3)

    frq_dsb = None
    assert FrequencyDoubleSideBand(183, 7, 2) != frq_dsb

    assert frq_dsb < FrequencyDoubleSideBand(183, 7, 2)

    assert FrequencyDoubleSideBand(182, 7, 2) < FrequencyDoubleSideBand(183, 7, 2)
    assert FrequencyDoubleSideBand(184, 7, 2) > FrequencyDoubleSideBand(183, 7, 2)


def test_frequency_double_side_band_channel_distances():
    """Test the frequency double side band object: get the distance between two bands."""
    frq_dsb = FrequencyDoubleSideBand(183, 7, 2)
    mydist = frq_dsb.distance(175.5)
    assert mydist == 0.5

    mydist = frq_dsb.distance(190.5)
    assert mydist == 0.5

    np.testing.assert_almost_equal(frq_dsb.distance(175.6), 0.4)
    np.testing.assert_almost_equal(frq_dsb.distance(190.1), 0.1)

    mydist = frq_dsb.distance(185)
    assert mydist == np.inf

    mydist = frq_dsb.distance((183, 7.0, 2))
    assert mydist == 0

    mydist = frq_dsb.distance((183, 7.0, 1))
    assert mydist == 0

    mydist = frq_dsb.distance(FrequencyDoubleSideBand(183, 7.0, 2))
    assert mydist == 0


def test_frequency_double_side_band_channel_containment():
    """Test the frequency double side band object: check if one band contains another."""
    frq_range = FrequencyDoubleSideBand(183, 7, 2)

    assert 175.5 in frq_range
    assert frq_range in FrequencyDoubleSideBand(183, 6.5, 3)
    assert frq_range not in FrequencyDoubleSideBand(183, 4, 2)

    with pytest.raises(NotImplementedError):
        assert frq_range in FrequencyDoubleSideBand(183, 6.5, 3, 'MHz')

    frq_range = None
    assert (frq_range in FrequencyDoubleSideBand(183, 3, 2)) is False

    assert '183' not in FrequencyDoubleSideBand(183, 3, 2)


def test_frequency_range_class_method_convert():
    """Test the frequency range object: test the class method convert."""
    frq_range = FrequencyRange(89, 2)

    res = frq_range.convert(89)
    assert res == 89

    res = frq_range.convert({'central': 89, 'bandwidth': 2})
    assert res == FrequencyRange(89, 2)


def test_frequency_range_class_method_str():
    """Test the frequency range object: test the band description."""
    frq_range1 = FrequencyRange(89, 2)
    frq_range2 = FrequencyRange(89000, 2000, 'MHz')

    assert str(frq_range1) == "central=89 GHz width=2 GHz"
    assert str(frq_range2) == "central=89000 MHz width=2000 MHz"


def test_frequency_range_channel_equality():
    """Test the frequency range object: check if two bands are 'equal'."""
    frqr = FrequencyRange(2, 1)
    assert frqr is not None
    assert 1.7 == frqr
    assert 1.2 != frqr
    assert frqr == (2, 1)

    assert frqr == (2, 1, 'GHz')


def test_frequency_range_channel_containment():
    """Test the frequency range object: channel containment."""
    frqr = FrequencyRange(2, 1)
    assert 1.7 in frqr
    assert 2.8 not in frqr

    with pytest.raises(NotImplementedError):
        assert frqr in FrequencyRange(89, 2, 'MHz')

    frqr = None
    assert (frqr in FrequencyRange(89, 2)) is False

    assert '89' not in FrequencyRange(89, 2)


def test_frequency_range_channel_distances():
    """Test the frequency range object: derive distances between bands."""
    frqr = FrequencyRange(190.0, 2)

    mydist = frqr.distance(FrequencyRange(190, 2))
    assert mydist == 0
    mydist = frqr.distance(FrequencyRange(189.5, 2))
    assert mydist == np.inf
    mydist = frqr.distance(189.5)
    assert mydist == 0.5
    mydist = frqr.distance(188.0)
    assert mydist == np.inf


def test_wavelength_range():
    """Test the wavelength range object."""
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

    wr = WavelengthRange(10.5, 11.5, 12.5)
    np.testing.assert_almost_equal(wr.distance(11.1), 0.4)


def test_wavelength_range_cf_roundtrip():
    """Test the wavelength range object roundtrip to cf."""
    wr = WavelengthRange(1, 2, 3)

    assert WavelengthRange.from_cf(wr.to_cf()) == wr
    assert WavelengthRange.from_cf([str(item) for item in wr]) == wr
