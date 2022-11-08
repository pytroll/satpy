"""Tests for the SGLI L1B backend."""
import dask.array as da
from dask.array.core import normalize_chunks
from xarray import DataArray, Dataset, open_dataset


def test_open_dataset():
    """Test open_dataset function."""
    from satpy.readers.sgli_l1b import SGLIBackend
    filename = "/home/a001673/data/satellite/gcom-c/GC1SG1_202002231142M25511_1BSG_VNRDL_1008.h5"
    res = open_dataset(filename, engine=SGLIBackend, chunks={})
    assert isinstance(res, Dataset)
    data_array = res["Lt_VN01"]
    assert isinstance(data_array, DataArray)
    assert isinstance(data_array.data, da.Array)
    assert data_array.chunks == normalize_chunks((116, 157), data_array.shape)
