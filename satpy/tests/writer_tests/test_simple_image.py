"""Tests for the simple image writer."""
import unittest


class TestPillowWriter(unittest.TestCase):
    """Test Pillow/PIL writer."""

    def setUp(self):
        """Create temporary directory to save files to."""
        import tempfile
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Remove the temporary directory created for a test."""
        try:
            import shutil
            shutil.rmtree(self.base_dir, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _get_test_datasets():
        """Create DataArray for testing."""
        import datetime as dt

        import dask.array as da
        import xarray as xr
        ds1 = xr.DataArray(
            da.arange(100 * 200).reshape((100, 200)).rechunk(50),
            dims=("y", "x"),
            attrs={"name": "test",
                   "start_time": dt.datetime.now(dt.timezone.utc)}
        )
        return [ds1]

    def test_init(self):
        """Test creating the default writer."""
        from satpy.writers.simple_image import PillowWriter
        PillowWriter()

    def test_simple_write(self):
        """Test writing datasets with default behavior."""
        from satpy.writers.simple_image import PillowWriter
        datasets = self._get_test_datasets()
        w = PillowWriter(base_dir=self.base_dir)
        w.save_datasets(datasets)

    def test_simple_delayed_write(self):
        """Test writing datasets with delayed computation."""
        import dask.array as da
        from dask.delayed import Delayed

        from satpy.writers.core.compute import compute_writer_results
        from satpy.writers.simple_image import PillowWriter
        datasets = self._get_test_datasets()
        w = PillowWriter(base_dir=self.base_dir)
        res = w.save_datasets(datasets, compute=False)
        for r__ in res:
            # trollimage 1.27.0+ returns Arrays
            # trollimage <1.27.0 returns Delayed objects
            assert isinstance(r__, (Delayed, da.Array))
            r__.compute()
        compute_writer_results([res])
