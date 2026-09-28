"""Module for testing the satpy.readers.core.netcdf module."""

import itertools
from contextlib import nullcontext
from unittest import mock

import numpy as np
import pytest

from satpy.readers.core.netcdf import (
    OPEN_STRATEGIES,
    NetCDF4FileHandler,
    NetCDF4FsspecFileHandler,
    _FileHandleOpener,
)

# NOTE:
# The following fixtures are not defined in this file, but are used and injected by Pytest:
# - tmp_path
# - tmp_path_factory


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

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    def test_all_basic(self, netcdf_file, strategy, engine):
        """Test everything about the NetCDF4 class."""
        import xarray as xr

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)

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
        assert file_handler["ds2_sc"] == 42

    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    def test_caching(self, netcdf_file, strategy):
        """Test that variables smaller than cache_var_size are kept in memory once they are read."""
        h = NetCDF4FileHandler(netcdf_file, {}, {}, cache_var_size=1000, open_strategy=strategy)
        assert not h.cached_variables

        np.testing.assert_array_equal(h["ds2_s"], np.arange(10))
        assert h["ds2_sc"] == 42
        np.testing.assert_array_equal(h["test_group/ds1_i"],
                                      np.arange(10 * 100).reshape((10, 100)))
        np.testing.assert_array_equal(
                h["ds2_f"],
                np.arange(10. * 100).reshape((10, 100)))
        assert sorted(h.cached_variables.keys()) == ["ds2_s", "ds2_sc"]
        assert h["ds2_s"] is h.cached_variables["ds2_s"]

    @pytest.mark.parametrize(("strategy", "auto_maskandscale", "engine"), [
        params for params in itertools.product(OPEN_STRATEGIES, [False, True], ["netcdf4", "h5netcdf"])
        # h5netcdf can't mask and scale with the file_handle open strategy
        if params != ("file_handle", True, "h5netcdf")
    ])
    def test_cached_variables_match_uncached(self, cf_netcdf_file, strategy, auto_maskandscale, engine):
        """Test that variables cached with cache_var_size are read the same way as variables that aren't."""
        import xarray as xr

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
        # phony_dims is an option of the h5netcdf backend store only
        xarray_kwargs = {"phony_dims": "sort", "decode_times": False, "backend_kwargs": {"lock": False}}
        file_handler = NetCDF4FileHandler(cf_netcdf_file, {}, {}, engine=engine, xarray_kwargs=xarray_kwargs)

        assert file_handler._opener._store_open_kwargs == {"phony_dims": "sort", "lock": False}
        assert "phony_dims" not in file_handler._opener.open_dataset_kwargs
        assert "backend_kwargs" not in file_handler._opener.open_dataset_kwargs
        assert file_handler["time"].dtype == np.float64
        # the given kwargs are not modified
        assert xarray_kwargs == {"phony_dims": "sort", "decode_times": False, "backend_kwargs": {"lock": False}}

    def test_filenotfound(self):
        """Test that error is raised when file not found."""
        # NOTE: Some versions of NetCDF C report unknown file format on Windows
        with pytest.raises(IOError, match=".*(No such file or directory|Unknown file format).*"):
            NetCDF4FileHandler("/thisfiledoesnotexist.nc", {}, {})

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    @pytest.mark.parametrize(("var_name", "expected"), [
        ("test_group/ds1_f", np.arange(10. * 100, dtype=np.float32).reshape((10, 100))),
        ("ds2_s", np.arange(10, dtype=np.int8)),
        ("ds2_sc", np.int8(42)),
    ])
    def test_get_and_cache_npxr(self, netcdf_file, strategy, engine, var_name, expected):
        """Test that get_and_cache_npxr() reads variables into memory and keeps them."""
        import xarray as xr

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)

        data = file_handler.get_and_cache_npxr(var_name)
        assert isinstance(data, xr.DataArray)
        assert data.chunks is None
        assert data.dtype == expected.dtype
        np.testing.assert_array_equal(data.values, expected)
        # kept in memory: the file handle of the "file_handle" open strategy can't be read anymore once closed
        file_handler.close()
        assert file_handler.get_and_cache_npxr(var_name) is data

    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    def test_get_and_cache_npxr_for_other_keys(self, netcdf_file, strategy):
        """Test that get_and_cache_npxr() gives the same as the file handler for keys that aren't variables."""
        import xarray as xr

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy)

        assert file_handler.get_and_cache_npxr("ds2_f/attr/test_attr_str") == "test_string"
        group = file_handler.get_and_cache_npxr("test_group")
        assert isinstance(group, xr.Dataset)
        assert group["ds1_i"].shape == (10, 100)
        assert not file_handler.cached_variables

    def test_file_opened_once(self, netcdf_file):
        """Test that reading many variables only opens the file once and decodes each group once.

        Opening (and closing) the file for every variable gives each returned
        lazy array its own entry in xarray's global file cache. Once that LRU
        cache overflows it closes handles that sibling arrays are still reading
        through, which segfaults in libhdf5.

        """
        import xarray as xr
        from xarray.backends import NetCDF4DataStore

        with mock.patch.object(NetCDF4DataStore, "open", wraps=NetCDF4DataStore.open) as store_open, \
                mock.patch.object(xr, "open_dataset", wraps=xr.open_dataset) as open_dataset:
            file_handler = NetCDF4FileHandler(netcdf_file, {}, {})
            for var in ("test_group/ds1_f", "test_group/ds1_i", "ds2_f", "ds2_i", "ds2_s"):
                for _ in range(3):
                    file_handler[var]

        store_open.assert_called_once()
        groups = [call.args[0]._group for call in open_dataset.call_args_list]
        assert sorted(groups, key=str) == [None, "test_group"]

    @pytest.mark.parametrize("strategy", OPEN_STRATEGIES)
    @pytest.mark.parametrize("key", ["ds2_f", "test_group"])
    def test_attrs_are_not_shared(self, netcdf_file, key, strategy):
        """Test that modifying a returned variable or group does not affect later reads."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy)
        first = file_handler[key]
        first.attrs["test_attr_str"] = "modified"
        first.attrs["extra_attr"] = "added"

        second = file_handler[key]
        assert second.attrs["test_attr_str"] == "test_string"
        assert "extra_attr" not in second.attrs

    @pytest.mark.parametrize("engine", ["netcdf4", "h5netcdf"])
    @pytest.mark.parametrize(("strategy", "exp_new_cache_entries"), [
        ("shared_store", 1),
        ("file_handle", 0),
    ])
    def test_open_strategies(self, netcdf_file, engine, strategy, exp_new_cache_entries):
        """Test that every open strategy reads the same data and holds the expected number of files open."""
        from xarray.backends.file_manager import FILE_CACHE

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
        with pytest.raises(ValueError, match="Unknown open_strategy"):
            NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy="magic")

    @pytest.mark.parametrize(("cache_handle", "open_strategy", "exp_strategy"), [
        (True, None, "file_handle"),
        (True, "file_handle", "file_handle"),
        (True, "shared_store", None),
        (False, None, "shared_store"),
        (False, "shared_store", "shared_store"),
        (False, "file_handle", None),
    ])
    def test_cache_handle_deprecated(self, netcdf_file, cache_handle, open_strategy, exp_strategy):
        """Test that cache_handle is deprecated and mapped to an open strategy, which open_strategy can't contradict.

        ``exp_strategy`` is None when the two conflict.

        """
        if exp_strategy is None:
            conflict = pytest.raises(ValueError, match="conflicts with open_strategy")
        else:
            conflict = nullcontext()
        with pytest.warns(DeprecationWarning, match="cache_handle"), conflict:
            file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, cache_handle=cache_handle,
                                              open_strategy=open_strategy)
        if exp_strategy is not None:
            assert isinstance(file_handler._opener, _FileHandleOpener) == (exp_strategy == "file_handle")

    def test_file_handle_h5netcdf_maskandscale_raises(self, netcdf_file):
        """Test that the file_handle strategy refuses auto_maskandscale=True with h5netcdf, which can't apply it."""
        with pytest.raises(ValueError, match="can't apply auto_maskandscale=True"):
            NetCDF4FileHandler(netcdf_file, {}, {}, engine="h5netcdf", open_strategy="file_handle",
                               auto_maskandscale=True)


def _variable_keys(name, attr_names=("test_attr_str", "test_attr_int", "test_attr_float")):
    """Get the keys of variable ``name`` in the file content: its own, its properties' and its attributes'."""
    return [name, *[f"{name}/{prop}" for prop in ("dtype", "shape", "dimensions")],
            *[f"{name}/attr/{attr_name}" for attr_name in attr_names]]


EXPECTED_FILE_CONTENT_KEYS = [
    "test_group",
    "test_group/attr/test_attr_str",
    "test_group/attr/test_attr_int",
    "test_group/attr/test_attr_float",
    *_variable_keys("test_group/ds1_f"),
    *_variable_keys("test_group/ds1_i"),
    *_variable_keys("ds2_f"),
    *_variable_keys("ds2_i"),
    *_variable_keys("ds2_s", attr_names=()),
    *_variable_keys("ds2_sc", attr_names=()),
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

    def test_walked_attribute_values(self, netcdf_file, strategy, engine):
        """Test that walking through the file gives the attributes of the root, groups, and variables."""
        attrs = {"test_attr_str": "test_string", "test_attr_int": 0, "test_attr_float": 1.2}
        # "" is the root group
        expected = {f"{name}/attr/{attr}": value
                    for name in ("test_group", "test_group/ds1_f", "test_group/ds1_i", "ds2_f", "ds2_i", "")
                    for attr, value in attrs.items()}
        expected["/attr/test_attr_str_arr"] = "test_string2"

        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        walked = {key: val for key, val in file_handler.file_content.items() if "/attr/" in key}
        assert walked == expected
        # the same as looking them up one by one
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine)
        assert {key: file_handler[key] for key in expected} == expected

    def test_attributes_named_like_python_attributes(self, tmp_path, strategy, engine):
        """Test that attributes named like python attributes of the netCDF4 objects are read from the file."""
        from netCDF4 import Dataset
        filename = tmp_path / "test_attr_names.nc"
        with Dataset(filename, "w") as nc:
            nc.createDimension("x", 3)
            var = nc.createVariable("var", np.float32, ("x",))
            var.setncatts({"name": "var_long_name", "shape": "flat"})
            nc.setncattr("path", "/some/path")
        expected = {"var/attr/name": "var_long_name", "var/attr/shape": "flat", "/attr/path": "/some/path"}

        file_handler = NetCDF4FileHandler(filename, {}, {}, open_strategy=strategy, engine=engine)
        assert {key: file_handler[key] for key in expected} == expected
        file_handler = NetCDF4FileHandler(filename, {}, {}, open_strategy=strategy, engine=engine)
        assert {key: val for key, val in file_handler.file_content.items() if "/attr/" in key} == expected

    def test_root_variable_named_attr(self, tmp_path, strategy, engine):
        """Test that a root variable named "attr" is looked up as it is walked, without hiding the global attributes."""
        from netCDF4 import Dataset
        filename = tmp_path / "test_attr_variable.nc"
        with Dataset(filename, "w") as nc:
            nc.createDimension("x", 3)
            nc.createVariable("attr", np.float32, ("x",)).units = "K"
            nc.units = "global"
        expected = {"attr/shape": (3,), "attr/attr/units": "K", "/attr/units": "global"}

        file_handler = NetCDF4FileHandler(filename, {}, {}, open_strategy=strategy, engine=engine)
        assert {key: file_handler[key] for key in expected} == expected
        file_handler = NetCDF4FileHandler(filename, {}, {}, open_strategy=strategy, engine=engine)
        walked = dict(file_handler.file_content.items())
        assert {key: walked[key] for key in expected} == expected

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

    @pytest.mark.parametrize(("required", "replacements", "expected_keys"), [
        # attributes of the root group and of a group
        (["test_group/attr/test_attr_str", "attr/test_attr_str"], None,
         ["test_group/attr/test_attr_str", "attr/test_attr_str"]),
        # variables of the root group and of a group with their properties and attributes, without missing ones
        (["attr/test_attr_str", "ds2_i", "test_group/ds1_f", "fake_ds"], None,
         ["attr/test_attr_str", *_variable_keys("ds2_i"), *_variable_keys("test_group/ds1_f")]),
        # names composed with the replacements
        (["test_group/{some_parameter}/attr/test_attr_str", "test_group/attr/test_attr_str"],
         {"some_parameter": ["ds1_f", "ds1_i"], "another_parameter": ["not_used"]},
         ["test_group/ds1_f/attr/test_attr_str", "test_group/ds1_i/attr/test_attr_str",
          "test_group/attr/test_attr_str"]),
    ])
    def test_listed_keys(self, netcdf_file, strategy, engine, required, replacements, expected_keys):
        """Test that iterating is limited to the listed keys, but every key can be looked up."""
        filetype_info = {"required_netcdf_variables": required, "variable_name_replacements": replacements}
        file_handler = NetCDF4FileHandler(netcdf_file, {}, filetype_info, open_strategy=strategy, engine=engine)
        file_content = file_handler.file_content

        assert list(file_content) == expected_keys
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
            file_handle = file_handler._opener.get_metadata_root()
            handler_ref = weakref.ref(file_handler)
            del file_handler
            assert handler_ref() is None
        finally:
            gc.enable()
        assert len(FILE_CACHE) == num_cached_before
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


def _has_open_file(opener):
    """Tell if ``opener`` has opened the file, as a file handle or as an xarray backend store."""
    return getattr(opener, "file_handle", None) is not None or opener._root_store is not None


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
        assert not _has_open_file(file_handler._opener)
        assert len(FILE_CACHE) == num_cached_before

        assert file_handler["/attr/test_attr_str"] == "test_string"
        assert file_handler.accessor.engine == ("netcdf4" if isinstance(engine, list) else engine)
        assert _has_open_file(file_handler._opener)
        np.testing.assert_array_equal(file_handler["ds2_s"], np.arange(10))

    def test_accessor_opens_only_to_choose_engine(self, netcdf_file, strategy, engine):
        """Test that getting the accessor only opens the file to choose between several engines."""
        file_handler = NetCDF4FileHandler(netcdf_file, {}, {}, open_strategy=strategy, engine=engine,
                                          defer_open=True)

        assert file_handler.accessor.engine == ("netcdf4" if isinstance(engine, list) else engine)
        assert _has_open_file(file_handler._opener) == isinstance(engine, list)

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
        assert not _has_open_file(file_handler._opener)


class TestNetCDF4FsspecFileHandler:
    """Test the remote reading class."""

    def test_default_to_netcdf4_lib(self, tmp_path):
        """Test that the NetCDF4 backend is used by default."""
        import h5py

        # Create an empty HDF5
        fname = tmp_path / "test.nc"
        with h5py.File(fname, "w"):
            pass

        fh = NetCDF4FsspecFileHandler(fname, {}, {})
        assert fh.accessor.engine == "netcdf4"

    @pytest.mark.parametrize(("open_strategy", "nc4_opener", "h5_opener"), [
        ("shared_store", "xarray.backends.NetCDF4DataStore.open", "xarray.backends.H5NetCDFStore.open"),
        ("file_handle", "netCDF4.Dataset", "h5netcdf.File"),
    ])
    def test_use_h5netcdf_for_file_not_accessible_locally(self, open_strategy, nc4_opener, h5_opener):
        """Test that h5netcdf is used for files that are not accesible locally."""
        fname = "s3://bucket/object.nc"

        with mock.patch(nc4_opener, side_effect=OSError("not a local file")), mock.patch(h5_opener) as h5_open:
            with mock.patch("satpy.readers.core.netcdf.open_file_or_filename"):
                fh = NetCDF4FsspecFileHandler(fname, {}, {}, open_strategy=open_strategy)
                h5_open.assert_called()
                assert fh.accessor.engine == "h5netcdf"
