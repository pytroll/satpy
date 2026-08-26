"""Reader for AMSR2 L2 files in HDF5 format."""

from satpy.readers.amsr2_l1b import AMSR2L1BFileHandler


class AMSR2L2FileHandler(AMSR2L1BFileHandler):
    """AMSR2 level 2 file handler."""

    def mask_dataset(self, ds_info, data):
        """Mask data with the fill value."""
        fill_value = ds_info.get("fill_value", 65535)
        return data.where(data != fill_value)

    def scale_dataset(self, var_path, data):
        """Scale data with the scale factor attribute."""
        return data * self[var_path + "/attr/SCALE FACTOR"]

    def get_dataset(self, ds_id, ds_info):
        """Get output data and metadata of specified dataset."""
        var_path = ds_info["file_key"]

        data = self[var_path].squeeze()
        data = self.mask_dataset(ds_info, data)
        data = self.scale_dataset(var_path, data)

        if ds_info.get("name") == "ssw":
            data = data.rename({"dim_0": "y", "dim_1": "x"})
        metadata = self.get_metadata(ds_id, ds_info)
        data.attrs.update(metadata)
        return data
