"""Module for testing the satpy.readers.core.netcdf module."""

import os

import numpy as np
import pytest

from satpy.readers.core.netcdf import OPEN_STRATEGIES, NetCDF4FileHandler


class FakeNetCDF4FileHandler(NetCDF4FileHandler):
    """Swap-in NetCDF4 File Handler for reader tests to use."""

    def __init__(self, filename, filename_info, filetype_info, *args, extra_file_content=None, **kwargs):
        """Get fake file content from 'get_test_content'."""
        # The file is never opened: everything is read from the fake file
        # content, and the accessor of a single engine is known without opening.
        super().__init__(filename, filename_info, filetype_info, *args, defer_open=True, **kwargs)
        self.file_content = self.get_test_content(filename, filename_info, filetype_info)
        if extra_file_content:
            self.file_content.update(extra_file_content)

    def get_test_content(self, filename, filename_info, filetype_info):
        """Mimic reader input file content.

        Args:
            filename (str): input filename
            filename_info (dict): Dict of metadata pulled from filename
            filetype_info (dict): Dict of metadata from the reader's yaml config for this file type

        Returns: dict of file content with keys like:

            - 'dataset'
            - '/attr/global_attr'
            - 'dataset/attr/global_attr'
            - 'dataset/shape'
            - 'dataset/dimensions'
            - '/dimension/my_dim'

        """
        raise NotImplementedError("Fake File Handler subclass must implement 'get_test_content'")


@pytest.fixture(scope="module")
def netcdf_file(tmp_path_factory):
    """Create a test NetCDF4 file."""
    from netCDF4 import Dataset
    filename = tmp_path_factory.mktemp("data") / "test.nc"
    with Dataset(filename, "w") as nc:
        # Create dimensions
        nc.createDimension("rows", 10)
        nc.createDimension("cols", 100)

        # Create Group
        g1 = nc.createGroup("test_group")
        # Add datasets
        ds1_f = g1.createVariable("ds1_f", np.float32,
                                  dimensions=("rows", "cols"))
        ds1_f[:] = np.arange(10. * 100).reshape((10, 100))
        ds1_f.set_auto_scale(True)

        ds1_i = g1.createVariable("ds1_i", np.int32,
                                  dimensions=("rows", "cols"))
        ds1_i[:] = np.arange(10 * 100).reshape((10, 100))

        ds2_f = nc.createVariable("ds2_f", np.float32,
                                  dimensions=("rows", "cols"))
        ds2_f[:] = np.arange(10. * 100).reshape((10, 100))
        ds2_i = nc.createVariable("ds2_i", np.int32,
                                  dimensions=("rows", "cols"))
        ds2_i[:] = np.arange(10 * 100).reshape((10, 100))
        ds2_s = nc.createVariable("ds2_s", np.int8,
                                  dimensions=("rows",))
        ds2_s[:] = np.arange(10)
        ds2_sc = nc.createVariable("ds2_sc", np.int8, dimensions=())
        ds2_sc[:] = np.int8(42)

        # Add attributes
        nc.test_attr_str = "test_string"
        nc.test_attr_int = 0
        nc.test_attr_float = 1.2
        nc.test_attr_str_arr = np.array(b"test_string2")
        g1.test_attr_str = "test_string"
        g1.test_attr_int = 0
        g1.test_attr_float = 1.2
        for d in [ds1_f, ds1_i, ds2_f, ds2_i]:
            d.test_attr_str = "test_string"
            d.test_attr_int = 0
            d.test_attr_float = 1.2
    return filename


@pytest.fixture(scope="module")
def cf_netcdf_file(tmp_path_factory):
    """Create a test NetCDF4 file with CF encoded variables (scaling, fill value, coordinates, times)."""
    from netCDF4 import Dataset
    filename = tmp_path_factory.mktemp("data") / "test_cf.nc"
    with Dataset(filename, "w") as nc:
        nc.createDimension("x", 4)
        x = nc.createVariable("x", np.float32, ("x",))
        x[:] = np.arange(4)
        x.units = "m"
        scaled = nc.createVariable("scaled", np.int16, ("x",), fill_value=-1)
        scaled.scale_factor = np.float32(0.5)
        scaled.add_offset = np.float32(10.)
        scaled.units = "K"
        scaled.coordinates = "x"
        scaled[:] = np.ma.masked_array([1, 2, 3, 4], mask=[0, 0, 1, 0])
        time = nc.createVariable("time", np.float64, ("x",))
        time.units = "seconds since 2000-01-01"
        time[:] = np.arange(4)
    return filename


class TestNetCDF4FileHandler:
    """Test NetCDF4 File Handler Utility class."""

    def test_all_basic(self, netcdf_file):
        """Test everything about the NetCDF4 class."""
        import xarray as xr

        from satpy.readers.core.netcdf import NetCDF4FileHandler
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})

        assert file_handler["/dimension/rows"] == 10
        assert file_handler["/dimension/cols"] == 100

        for ds in ("test_group/ds1_f", "test_group/ds1_i", "ds2_f", "ds2_i"):
            assert file_handler[ds].dtype == (np.float32 if ds.endswith("f") else np.int32)
            assert file_handler[ds + "/shape"] == (10, 100)
            assert file_handler[ds + "/dimensions"] == ("rows", "cols")
            assert file_handler[ds + "/attr/test_attr_str"] == "test_string"
            assert file_handler[ds + "/attr/test_attr_int"] == 0
            assert file_handler[ds + "/attr/test_attr_float"] == 1.2

        test_group = file_handler["test_group"]
        assert test_group["ds1_i"].shape == (10, 100)
        assert test_group["ds1_i"].dims == ("rows", "cols")

        assert file_handler["/attr/test_attr_str"] == "test_string"
        assert file_handler["/attr/test_attr_str_arr"] == "test_string2"
        assert file_handler["/attr/test_attr_int"] == 0
        assert file_handler["/attr/test_attr_float"] == 1.2

        global_attrs = {
            "test_attr_str": "test_string",
            "test_attr_str_arr": "test_string2",
            "test_attr_int": 0,
            "test_attr_float": 1.2
            }
        assert file_handler["/attrs"] == global_attrs

        assert isinstance(file_handler.get("ds2_f")[:], xr.DataArray)
        assert file_handler.get("fake_ds") is None
        assert file_handler.get("fake_ds", "test") == "test"

        assert ("ds2_f" in file_handler) is True
        assert ("fake_ds" in file_handler) is False
        assert file_handler._opener.file_handle is None
        assert file_handler["ds2_sc"] == 42

    def test_listed_variables(self, netcdf_file):
        """Test that only listed variables/attributes area collected."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        filetype_info = {
            "required_netcdf_variables": [
                "test_group/attr/test_attr_str",
                "attr/test_attr_str",
            ]
        }
        file_handler = NetCDF4FileHandler(netcdf_file, {}, filetype_info)
        assert len(file_handler.file_content) == 2
        assert "test_group/attr/test_attr_str" in file_handler.file_content
        assert "attr/test_attr_str" in file_handler.file_content

    @pytest.mark.parametrize(
        ("required_variables", "expected_variable"),
        [
            (["ds2_f"], "ds2_f"),
            (["attr/test_attr_str", "ds2_i"], "ds2_i"),
        ],
    )
    def test_listed_root_variables(self, netcdf_file, required_variables, expected_variable):
        """Test collection of required variables located at the NetCDF root."""
        filetype_info = {"required_netcdf_variables": required_variables}

        file_handler = NetCDF4FileHandler(netcdf_file, {}, filetype_info)

        assert expected_variable in file_handler.file_content
        assert file_handler.file_content[expected_variable + "/shape"] == (10, 100)

    def test_listed_variables_with_composing(self, netcdf_file):
        """Test that composing for listed variables is performed."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        filetype_info = {
            "required_netcdf_variables": [
                "test_group/{some_parameter}/attr/test_attr_str",
                "test_group/attr/test_attr_str",
            ],
            "variable_name_replacements": {
                "some_parameter": [
                    "ds1_f",
                    "ds1_i",
                ],
                "another_parameter": [
                    "not_used"
                ],
            }
        }
        file_handler = NetCDF4FileHandler(netcdf_file, {}, filetype_info)
        assert len(file_handler.file_content) == 3
        assert "test_group/ds1_f/attr/test_attr_str" in file_handler.file_content
        assert "test_group/ds1_i/attr/test_attr_str" in file_handler.file_content
        assert not any("not_used" in var for var in file_handler.file_content)
        assert not any("some_parameter" in var for var in file_handler.file_content)
        assert not any("another_parameter" in var for var in file_handler.file_content)
        assert "test_group/attr/test_attr_str" in file_handler.file_content

    def test_caching(self, netcdf_file):
        """Test that caching works as intended."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler
        h = NetCDF4FileHandler(netcdf_file, {}, {}, cache_var_size=1000,
                               open_strategy="file_handle")
        assert h._opener.file_handle is not None
        assert h._opener.file_handle.isopen()
        # variables are only cached when they are accessed, without walking through the file
        assert not h.cached_variables
        assert h.file_content._file_keys is None

        # with caching, these tests access different lines than without
        np.testing.assert_array_equal(h["ds2_s"], np.arange(10))
        assert h["ds2_sc"] == 42
        np.testing.assert_array_equal(h["test_group/ds1_i"],
                                      np.arange(10 * 100).reshape((10, 100)))
        # check that root variables can still be read from cached file object,
        # even if not cached themselves
        np.testing.assert_array_equal(
                h["ds2_f"],
                np.arange(10. * 100).reshape((10, 100)))
        assert sorted(h.cached_variables.keys()) == ["ds2_s", "ds2_sc"]
        assert h["ds2_s"] is h.cached_variables["ds2_s"]
        h.__del__()
        assert not h._opener.file_handle.isopen()

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize("auto_maskandscale", [False, True])
    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    def test_cached_variables_match_uncached(self, cf_netcdf_file, strategy, auto_maskandscale, engine):
        """Test that variables cached with cache_var_size are read the same way as variables that aren't."""
        import xarray as xr

        from satpy.readers.core.netcdf import NetCDF4FileHandler

        if strategy == "file_handle" and auto_maskandscale and engine == "h5netcdf":
            pytest.skip("h5netcdf can't mask and scale with the file_handle open strategy")
        kwargs = {"open_strategy": strategy, "auto_maskandscale": auto_maskandscale, "engine": engine}
        cached_handler = NetCDF4FileHandler(cf_netcdf_file, {}, {}, cache_var_size=1000, **kwargs)
        uncached_handler = NetCDF4FileHandler(cf_netcdf_file, {}, {}, **kwargs)

        for var_name in ("scaled", "time"):
            cached = cached_handler[var_name]
            uncached = uncached_handler[var_name]
            assert var_name in cached_handler.cached_variables
            assert var_name not in uncached_handler.cached_variables
            assert cached.chunks is None
            xr.testing.assert_identical(cached, uncached.compute())

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf", ["netcdf4", "h5netcdf"]])
    def test_xarray_kwargs_split_between_store_and_open_dataset(self, cf_netcdf_file, engine):
        """Test that xarray_kwargs are given to the backend store or xarray.open_dataset depending on the option."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        # phony_dims is an option of the h5netcdf backend store only
        xarray_kwargs = {"phony_dims": "sort", "decode_times": False, "backend_kwargs": {"lock": False}}
        file_handler = NetCDF4FileHandler(cf_netcdf_file, {}, {}, engine=engine, xarray_kwargs=xarray_kwargs)

        assert file_handler._opener._store_open_kwargs == {"phony_dims": "sort", "lock": False}
        assert "phony_dims" not in file_handler._open_dataset_kwargs
        assert "backend_kwargs" not in file_handler._open_dataset_kwargs
        assert file_handler["time"].dtype == np.float64
        # the given kwargs are not modified
        assert xarray_kwargs == {"phony_dims": "sort", "decode_times": False, "backend_kwargs": {"lock": False}}

    def test_filenotfound(self):
        """Test that error is raised when file not found."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        # NOTE: Some versions of NetCDF C report unknown file format on Windows
        with pytest.raises(IOError, match=".*(No such file or directory|Unknown file format).*"):
            NetCDF4FileHandler("/thisfiledoesnotexist.nc", {}, {})

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    def test_get_and_cache_npxr_is_xr(self, netcdf_file, engine):
        """Test that get_and_cache_npxr() returns xr.DataArray."""
        import xarray as xr

        from satpy.readers.core.netcdf import NetCDF4FileHandler
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy="file_handle", engine=engine)

        data = file_handler.get_and_cache_npxr("test_group/ds1_f")
        assert isinstance(data, xr.DataArray)

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    def test_get_and_cache_npxr_for_scalar(self, netcdf_file, engine):
        """Test that get_and_cache_npxr() returns xr.DataArray."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy="file_handle", engine=engine)

        data = file_handler.get_and_cache_npxr("ds2_sc")
        # WARN: h5netcdf returns an int64!
        assert data.dtype in [np.int8, np.int64], "Scalar should be of type int8"
        assert data == 42

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    def test_get_and_cache_npxr_data_is_cached(self, netcdf_file, engine):
        """Test that the data are cached when get_and_cache_npxr() is called."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy="file_handle", engine=engine)
        data = file_handler.get_and_cache_npxr("test_group/ds1_f")

        # The file handle can't be read anymore once closed, the data have to come from the cache
        file_handler.close()
        assert file_handler.get_and_cache_npxr("test_group/ds1_f") is data

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize(("var_name", "expected"), [
        ("test_group/ds1_f", np.arange(10. * 100).reshape((10, 100))),
        ("ds2_s", np.arange(10)),
        ("ds2_sc", 42),
    ])
    def test_get_and_cache_npxr_without_file_handle(self, netcdf_file, engine, var_name, expected):
        """Test that get_and_cache_npxr() reads variables after the file handle was closed.

        With any open strategy but "file_handle" the variable objects
        collected in ``__init__`` belong to a closed file, so the data has to
        be read through xarray instead.
        """
        import xarray as xr

        from satpy.readers.core.netcdf import NetCDF4FileHandler
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, engine=engine)

        data = file_handler.get_and_cache_npxr(var_name)
        assert isinstance(data, xr.DataArray)
        assert data.chunks is None
        np.testing.assert_array_equal(data.values, expected)
        assert var_name in file_handler.cached_variables

    def test_file_opened_once(self, netcdf_file, monkeypatch):
        """Test that reading many variables only opens the file once and decodes each group once.

        Opening (and closing) the file for every variable gives each returned
        lazy array its own entry in xarray's global file cache. Once that LRU
        cache overflows it closes handles that sibling arrays are still reading
        through, which segfaults in libhdf5.

        """
        import xarray as xr
        from xarray.backends import NetCDF4DataStore

        from satpy.readers.core.netcdf import NetCDF4FileHandler

        store_opens = []
        real_store_open = NetCDF4DataStore.open

        def counting_store_open(*args, **kwargs):
            store = real_store_open(*args, **kwargs)
            store_opens.append(store)
            return store

        groups = []
        real_open_dataset = xr.open_dataset

        def counting_open_dataset(store, **kwargs):
            groups.append(store._group)
            return real_open_dataset(store, **kwargs)

        monkeypatch.setattr(NetCDF4DataStore, "open", counting_store_open)
        monkeypatch.setattr(xr, "open_dataset", counting_open_dataset)
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
        for var in ("test_group/ds1_f", "test_group/ds1_i", "ds2_f", "ds2_i", "ds2_s"):
            for _ in range(3):
                file_handler[var]

        assert len(store_opens) == 1
        assert sorted(groups, key=str) == [None, "test_group"]

    def test_file_cache_does_not_grow_with_variables(self, netcdf_file):
        """Test that repeated variable access does not fill xarray's file cache."""
        from xarray.backends.file_manager import FILE_CACHE

        from satpy.readers.core.netcdf import NetCDF4FileHandler

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
        file_handler["ds2_f"]
        num_cached = len(FILE_CACHE)
        for _ in range(20):
            file_handler["ds2_f"]
            file_handler["ds2_i"]

        assert len(FILE_CACHE) == num_cached

    def test_variable_attrs_are_not_shared(self, netcdf_file):
        """Test that modifying a returned variable does not affect later reads."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
        first = file_handler["ds2_f"]
        first.attrs["test_attr_str"] = "modified"
        first.attrs["extra_attr"] = "added"

        second = file_handler["ds2_f"]
        assert second.attrs["test_attr_str"] == "test_string"
        assert "extra_attr" not in second.attrs

    def test_close_releases_open_datasets(self, netcdf_file):
        """Test that close() releases the datasets held open for reading."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
        file_handler["ds2_f"]
        assert file_handler._opener._datasets

        file_handler.close()

        assert not file_handler._opener._datasets
        # the variable is still readable, the file is simply reopened
        assert file_handler["ds2_f"].shape == (10, 100)

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize(("strategy", "exp_new_cache_entries"), [
        ("shared_store", 1),
        ("file_handle", 0),
    ])
    def test_open_strategies(self, netcdf_file, engine, strategy, exp_new_cache_entries):
        """Test that every open strategy reads the same data and holds the expected number of files open."""
        from xarray.backends.file_manager import FILE_CACHE

        from satpy.readers.core.netcdf import NetCDF4FileHandler

        num_cached_before = len(FILE_CACHE)
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, engine=engine, open_strategy=strategy)
        expected = np.arange(10. * 100).reshape((10, 100))
        for var_name in ("test_group/ds1_f", "ds2_f"):
            data = file_handler[var_name]
            assert data.dims == ("rows", "cols")
            assert data.attrs["test_attr_str"] == "test_string"
            np.testing.assert_array_equal(data.values, expected)
        # a second access must not open anything new
        file_handler["ds2_i"]
        assert len(FILE_CACHE) - num_cached_before == exp_new_cache_entries

        file_handler.close()
        assert len(FILE_CACHE) == num_cached_before
        assert not file_handler._opener._datasets
        assert file_handler._opener._root_store is None

    def test_invalid_open_strategy(self, netcdf_file):
        """Test that unknown open strategies are rejected."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        with pytest.raises(ValueError, match="Unknown open_strategy"):
            NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy="magic")

    @pytest.mark.parametrize(("cache_handle", "open_strategy", "exp_strategy"), [
        (True, None, "file_handle"),
        (True, "file_handle", "file_handle"),
        (False, None, "shared_store"),
    ])
    def test_cache_handle_deprecated(self, netcdf_file, cache_handle, open_strategy, exp_strategy):
        """Test that cache_handle is deprecated and mapped to an open strategy."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        with pytest.warns(DeprecationWarning, match="cache_handle"):
            file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, cache_handle=cache_handle,
                                              open_strategy=open_strategy)
        assert (file_handler._opener.file_handle is not None) == (exp_strategy == "file_handle")

    def test_cache_handle_conflicting_open_strategy(self, netcdf_file):
        """Test that cache_handle can't be combined with an open strategy it doesn't map to."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        with pytest.warns(DeprecationWarning, match="cache_handle"), \
                pytest.raises(ValueError, match="conflicts with open_strategy"):
            NetCDF4FileHandler(netcdf_file, {}, {}, cache_handle=True, open_strategy="shared_store")
        with pytest.warns(DeprecationWarning, match="cache_handle"), \
                pytest.raises(ValueError, match="conflicts with open_strategy"):
            NetCDF4FileHandler(netcdf_file, {}, {}, cache_handle=False, open_strategy="file_handle")

    def test_file_handle_h5netcdf_maskandscale_raises(self, netcdf_file):
        """Test that the file_handle strategy refuses to return unscaled data with h5netcdf."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        with pytest.raises(ValueError, match="can't apply auto_maskandscale=True"):
            NetCDF4FileHandler(netcdf_file, {}, {}, engine="h5netcdf", open_strategy="file_handle",
                               auto_maskandscale=True)

    def test_group_attrs_are_not_shared(self, netcdf_file):
        """Test that modifying a returned group does not affect later reads."""
        from satpy.readers.core.netcdf import NetCDF4FileHandler

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
        first = file_handler["test_group"]
        first.attrs["extra_attr"] = "added"

        second = file_handler["test_group"]
        assert "extra_attr" not in second.attrs

EXPECTED_FILE_CONTENT_KEYS = [
    "test_group",
    "test_group/attr/test_attr_str",
    "test_group/attr/test_attr_int",
    "test_group/attr/test_attr_float",
    *[f"test_group/{var}{suffix}" for var in ("ds1_f", "ds1_i") for suffix in (
        "", "/dtype", "/shape", "/dimensions",
        "/attr/test_attr_str", "/attr/test_attr_int", "/attr/test_attr_float")],
    *[f"{var}{suffix}" for var in ("ds2_f", "ds2_i") for suffix in (
        "", "/dtype", "/shape", "/dimensions",
        "/attr/test_attr_str", "/attr/test_attr_int", "/attr/test_attr_float")],
    *[f"{var}{suffix}" for var in ("ds2_s", "ds2_sc") for suffix in ("", "/dtype", "/shape", "/dimensions")],
    "/attr/test_attr_str",
    "/attr/test_attr_int",
    "/attr/test_attr_float",
    "/attr/test_attr_str_arr",
    "/attrs",
    "/dimension/rows",
    "/dimension/cols",
]


@pytest.fixture(scope="module")
def nested_netcdf_file(tmp_path_factory):
    """Create a test NetCDF4 file with dimensions defined in (nested) groups."""
    from netCDF4 import Dataset
    filename = tmp_path_factory.mktemp("data") / "test_nested.nc"
    with Dataset(filename, "w") as nc:
        nc.createDimension("root_dim", 2)
        group = nc.createGroup("group")
        group.createDimension("group_dim", 3)
        subgroup = group.createGroup("subgroup")
        subgroup.createDimension("subgroup_dim", 4)
        var = subgroup.createVariable("var", np.float32, ("root_dim", "group_dim", "subgroup_dim"))
        var[:] = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        var.units = "K"
    return filename


@pytest.fixture(scope="module")
def unlimited_netcdf_file(tmp_path_factory):
    """Create a test NetCDF4 file with unlimited dimensions in the root group and in a group."""
    from netCDF4 import Dataset
    filename = tmp_path_factory.mktemp("data") / "test_unlimited.nc"
    with Dataset(filename, "w") as nc:
        nc.createDimension("root_dim", None)
        group = nc.createGroup("group")
        group.createDimension("group_dim", None)
        var = group.createVariable("var", np.float32, ("root_dim", "group_dim"))
        var[:] = np.arange(3 * 5).reshape((3, 5))
    return filename


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
@pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
class TestNetCDF4FileContent:
    """Test the lazy file content mapping of the NetCDF4 file handler."""

    def test_keys_and_order(self, netcdf_file, strategy, engine):
        """Test that iterating gives every key of the file, each variable before its attributes."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert list(file_content) == EXPECTED_FILE_CONTENT_KEYS
        assert len(file_content) == len(EXPECTED_FILE_CONTENT_KEYS)
        values = dict(file_content.items())
        assert file_handler.accessor.is_group(values["test_group"])
        for var_name in ("test_group/ds1_f", "ds2_f", "ds2_sc"):
            assert file_handler.accessor.is_variable(values[var_name])

    def test_values(self, netcdf_file, strategy, engine):
        """Test the values of the keys that aren't groups or variables."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert file_content["test_group/ds1_f/dtype"] == np.float32
        assert file_content["test_group/ds1_f/shape"] == (10, 100)
        assert file_content["test_group/ds1_f/dimensions"] == ("rows", "cols")
        assert file_content["ds2_sc/shape"] == ()
        assert file_content["test_group/ds1_i/attr/test_attr_float"] == 1.2
        assert file_content["test_group/attr/test_attr_int"] == 0
        assert file_content["/attr/test_attr_str_arr"] == "test_string2"
        # alias of the global attribute as listed in required_netcdf_variables
        assert file_content["attr/test_attr_str"] == "test_string"
        assert file_content["/attrs"] == {"test_attr_str": "test_string", "test_attr_int": 0,
                                          "test_attr_float": 1.2, "test_attr_str_arr": "test_string2"}
        assert file_content["/dimension/cols"] == 100

    @pytest.mark.parametrize("key", [
        "fake_ds",
        "",
        "/test_group",
        "test_group/fake_ds",
        "ds2_f/fake_property",
        "ds2_f/shape/extra",
        "ds2_f/attr/fake_attr",
        # attributes are not python attributes of the netCDF4 variable objects
        "ds2_f/attr/shape",
        "ds2_f/attr/name",
        "test_group/attr/fake_attr",
        "/attr/fake_attr",
        "/dimension/fake_dim",
        "test_group/dimension/rows",
        "ds2_f/dimension/rows",
        "test_group/shape",
    ])
    def test_missing_keys(self, netcdf_file, strategy, engine, key):
        """Test that keys that are not in the file raise a KeyError."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)

        assert key not in file_handler.file_content
        assert key not in file_handler
        with pytest.raises(KeyError):
            file_handler.file_content[key]
        assert file_handler.get(key) is None

    def test_group_dimensions(self, nested_netcdf_file, strategy, engine):
        """Test that the dimensions of every group are available."""
        file_handler = NetCDF4FileHandler(nested_netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert file_content["/dimension/root_dim"] == 2
        assert file_content["group/dimension/group_dim"] == 3
        assert file_content["group/subgroup/dimension/subgroup_dim"] == 4
        assert file_content["group/subgroup/var/shape"] == (2, 3, 4)
        assert list(file_content) == [
            "group",
            "group/subgroup",
            "group/subgroup/var",
            "group/subgroup/var/dtype",
            "group/subgroup/var/shape",
            "group/subgroup/var/dimensions",
            "group/subgroup/var/attr/units",
            "group/subgroup/dimension/subgroup_dim",
            "group/dimension/group_dim",
            "/attrs",
            "/dimension/root_dim",
        ]

    def test_unlimited_dimensions(self, unlimited_netcdf_file, strategy, engine):
        """Test that the size of unlimited dimensions is their current size."""
        file_handler = NetCDF4FileHandler(unlimited_netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert file_content["/dimension/root_dim"] == 3
        assert file_content["group/dimension/group_dim"] == 5
        assert file_content["group/var/shape"] == (3, 5)
        # the sizes found by walking through the file are the same
        file_handler = NetCDF4FileHandler(unlimited_netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        dimensions = {key: val for key, val in file_handler.file_content.items() if "/dimension/" in key}
        assert dimensions == {"/dimension/root_dim": 3, "group/dimension/group_dim": 5}

    def test_file_not_walked_for_lookups(self, netcdf_file, strategy, engine):
        """Test that looking keys up doesn't walk through the whole file."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert file_content._file_keys is None
        assert file_content["test_group/ds1_f/shape"] == (10, 100)
        assert "ds2_f" in file_content
        assert file_handler["ds2_f"].shape == (10, 100)
        assert file_content._file_keys is None
        assert len(file_content) == len(EXPECTED_FILE_CONTENT_KEYS)
        assert file_content._file_keys is not None

    def test_listed_keys(self, netcdf_file, strategy, engine):
        """Test that iterating is limited to the listed keys, but every key can be looked up."""
        filetype_info = {"required_netcdf_variables": ["attr/test_attr_str", "test_group/ds1_f", "fake_ds"]}
        file_handler = NetCDF4FileHandler(netcdf_file, {}, filetype_info, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert list(file_content) == [
            "attr/test_attr_str",
            "test_group/ds1_f",
            "test_group/ds1_f/dtype",
            "test_group/ds1_f/shape",
            "test_group/ds1_f/dimensions",
            "test_group/ds1_f/attr/test_attr_str",
            "test_group/ds1_f/attr/test_attr_int",
            "test_group/ds1_f/attr/test_attr_float",
        ]
        assert file_content["ds2_f/shape"] == (10, 100)

    def test_file_closed_when_handler_deleted(self, netcdf_file, strategy, engine):
        """Test that the file content doesn't keep the file handler (and its file) alive."""
        import gc
        import weakref

        from xarray.backends.file_manager import FILE_CACHE

        num_cached_before = len(FILE_CACHE)
        gc.disable()
        try:
            file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
            file_content = file_handler.file_content
            assert len(file_content) == len(EXPECTED_FILE_CONTENT_KEYS)
            file_handle = file_handler._opener.file_handle
            handler_ref = weakref.ref(file_handler)
            del file_handler
            assert handler_ref() is None
        finally:
            gc.enable()
        assert len(FILE_CACHE) == num_cached_before
        if file_handle is not None:
            assert not _is_open(file_handle)

    def test_readable_after_close(self, netcdf_file, strategy, engine):
        """Test that the file content can still be read after closing the file handler if the file can be reopened."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        assert file_handler["ds2_f/attr/test_attr_str"] == "test_string"
        file_handler.close()

        # remembered
        assert file_handler["ds2_f/attr/test_attr_str"] == "test_string"
        if strategy == "file_handle":
            # the file handle can't be reopened
            return
        assert file_handler["test_group/ds1_i/attr/test_attr_int"] == 0
        assert file_handler.accessor.is_group(file_handler.file_content["test_group"])
        np.testing.assert_array_equal(file_handler["ds2_s"], np.arange(10))


def _is_open(file_handle):
    if hasattr(file_handle, "isopen"):
        return file_handle.isopen()
    return not file_handle._closed


@pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf", ["netcdf4", "h5netcdf"]])
@pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
class TestDeferOpen:
    """Test opening the file only when something is first read from it."""

    def test_not_opened_until_read(self, netcdf_file, strategy, engine):
        """Test that the file is opened on the first read."""
        from xarray.backends.file_manager import FILE_CACHE

        num_cached_before = len(FILE_CACHE)
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine,
                                          defer_open=True)
        assert file_handler._opener.file_handle is None
        assert file_handler._opener._root_store is None
        assert len(FILE_CACHE) == num_cached_before

        assert file_handler["/attr/test_attr_str"] == "test_string"
        assert file_handler.accessor.engine == ("netcdf4" if isinstance(engine, list) else engine)
        assert (file_handler._opener.file_handle is not None) == (strategy == "file_handle")
        np.testing.assert_array_equal(file_handler["ds2_s"], np.arange(10))

    def test_accessor_opens_only_to_choose_engine(self, netcdf_file, strategy, engine):
        """Test that getting the accessor only opens the file to choose between several engines."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine,
                                          defer_open=True)

        assert file_handler.accessor.engine == ("netcdf4" if isinstance(engine, list) else engine)
        is_open = file_handler._opener.file_handle is not None or file_handler._opener._root_store is not None
        assert is_open == isinstance(engine, list)

    def test_missing_file_fails_on_first_read(self, tmp_path, strategy, engine):
        """Test that a file that can't be opened only fails when it is first read."""
        file_handler = NetCDF4FileHandler(tmp_path / "missing.nc", {}, {}, open_strategy=strategy,
                                          engine=engine, defer_open=True)
        # engine lists raise a RuntimeError when none of them can open the file
        with pytest.raises((IOError, RuntimeError)):
            file_handler["/attr/test_attr_str"]
        with pytest.raises((IOError, RuntimeError)):
            "ds2_f" in file_handler  # noqa: B015

    def test_close_before_open(self, netcdf_file, strategy, engine):
        """Test that closing a file handler that never opened its file doesn't open it."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine,
                                          defer_open=True)
        file_handler.close()
        assert file_handler._opener.file_handle is None
        assert file_handler._opener._root_store is None


class TestNetCDF4FsspecFileHandler:
    """Test the remote reading class."""

    def test_default_to_netcdf4_lib(self):
        """Test that the NetCDF4 backend is used by default."""
        import tempfile

        import h5py

        from satpy.readers.core.netcdf import NetCDF4FsspecFileHandler

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty HDF5
            fname = os.path.join(tmpdir, "test.nc")
            fid = h5py.File(fname, "w")
            fid.close()

            fh = NetCDF4FsspecFileHandler(fname, {}, {})
            assert fh.accessor.engine == "netcdf4"

    @pytest.mark.parametrize(("open_strategy", "h5_opener"), [
        ("shared_store", "xarray.backends.H5NetCDFStore.open"),
        ("file_handle", "h5netcdf.File"),
    ])
    def test_use_h5netcdf_for_file_not_accessible_locally(self, open_strategy, h5_opener):
        """Test that h5netcdf is used for files that are not accesible locally."""
        from unittest.mock import patch

        fname = "s3://bucket/object.nc"

        with patch(h5_opener) as h5_open:
            with patch("satpy.readers.core.netcdf.open_file_or_filename"):
                from satpy.readers.core.netcdf import NetCDF4FsspecFileHandler

                fh = NetCDF4FsspecFileHandler(fname, {}, {}, open_strategy=open_strategy)
                h5_open.assert_called()
                assert fh.accessor.engine == "h5netcdf"

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    def test_netcdf_engines(self, netcdf_file, engine):
        """Test that h5netcdf engine is used."""
        from satpy.readers.core.netcdf import NetCDF4FsspecFileHandler

        fh = NetCDF4FsspecFileHandler(netcdf_file, {}, {}, engine=engine)
        assert fh.accessor.engine == engine
        np.testing.assert_array_equal(
                fh["ds2_f"],
                np.arange(10. * 100).reshape((10, 100)))


NC_ATTRS = {
    "standard_name": "test_data",
    "scale_factor": 0.01,
    "add_offset": 0}

def test_get_data_as_xarray_netcdf4(tmp_path):
    """Test getting xr.DataArray from netcdf4 variable."""
    import numpy as np

    from satpy.readers.core.netcdf import get_data_as_xarray

    data = np.array([1, 2, 3])
    fname = tmp_path / "test.nc"
    dset = _write_test_netcdf4(fname, data)

    res = get_data_as_xarray(dset["test_data"])
    np.testing.assert_equal(res.data, data)
    assert res.attrs == NC_ATTRS


def test_get_data_as_xarray_scalar_netcdf4(tmp_path):
    """Test getting scalar xr.DataArray from netcdf4 variable."""
    import numpy as np

    from satpy.readers.core.netcdf import get_data_as_xarray

    data = 1
    fname = tmp_path / "test.nc"
    dset = _write_test_netcdf4(fname, data)

    res = get_data_as_xarray(dset["test_data"])
    np.testing.assert_equal(res.data, np.array(data))
    assert res.attrs == NC_ATTRS


def _write_test_netcdf4(fname, data):
    import netCDF4 as nc

    dset = nc.Dataset(fname, "w")
    try:
        dset.createDimension("y", data.size)
        dims = ("y",)
    except AttributeError:
        dims = ()
    var = dset.createVariable("test_data", "uint8", dims)
    var[:] = data
    var.setncatts(NC_ATTRS)
    # Turn off automatic scale factor and offset handling
    dset.set_auto_maskandscale(False)

    return dset


def test_get_data_as_xarray_h5netcdf(tmp_path):
    """Test getting xr.DataArray from h5netcdf variable."""
    import numpy as np

    from satpy.readers.core.netcdf import get_data_as_xarray

    data = np.array([1, 2, 3])
    fname = tmp_path / "test.nc"
    fid = _write_test_h5netcdf(fname, data)

    res = get_data_as_xarray(fid["test_data"])
    np.testing.assert_equal(res.data, data)
    assert res.attrs == NC_ATTRS


def _write_test_h5netcdf(fname, data):
    import h5netcdf

    fid = h5netcdf.File(fname, "w")
    try:
        fid.dimensions = {"y": data.size}
        dims = ("y",)
    except AttributeError:
        dims = ()
    var = fid.create_variable("test_data", dims, "uint8", data=data)
    for key in NC_ATTRS:
        var.attrs[key] = NC_ATTRS[key]

    return fid


def test_get_data_as_xarray_scalar_h5netcdf(tmp_path):
    """Test getting xr.DataArray from h5netcdf variable."""
    import numpy as np

    from satpy.readers.core.netcdf import get_data_as_xarray

    data = 1
    fname = tmp_path / "test.nc"
    fid = _write_test_h5netcdf(fname, data)

    res = get_data_as_xarray(fid["test_data"])
    np.testing.assert_equal(res.data, np.array(data))
    assert res.attrs == NC_ATTRS
