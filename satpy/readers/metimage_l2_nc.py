
"""EUMETSAT EPS-SG Visible/Infrared Imager (VII) Level 2 products reader."""

import logging

import xarray as xr

from satpy.readers.core.metimage_nc import METimageNCBaseFileHandler

logger = logging.getLogger(__name__)


class METimageL2NCFileHandler(METimageNCBaseFileHandler):
    """Reader class for VII L2 products in netCDF format."""

    def _perform_orthorectification(self, variable: xr.DataArray, orthorect_data_name: str) -> xr.DataArray:
        """Perform the orthorectification.

        Args:
            variable: DataArray containing the dataset to correct for orthorectification.
            orthorect_data_name: name of the orthorectification correction data in the product.

        Returns:
            array containing the corrected values and all the original metadata.

        """
        try:
            orthorect_data = self[orthorect_data_name]
            variable += orthorect_data
        except KeyError:
            logger.warning("Required dataset %s for orthorectification not available, skipping", orthorect_data_name)
        return variable
