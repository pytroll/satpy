
"""EUMETSAT EPS-SG Visible/Infrared Imager (VII) Level 2 products reader.

.. note::

   The orthorectification + clouds parallax correction is deactivated by default. If you want to use it,
   you need to activate it explicitly by setting the ``orthorect`` keyword argument to ``True``, e.g.:
      .. code-block:: python

        scn = Scene(filenames=filenames, reader='metimage_l2_nc', reader_kwargs={'orthorect': True})

    Note that the correction is not available for the CLD, WVV and WVI products.
"""

import logging

import xarray as xr

from satpy.readers.core.metimage_nc import METimageNCBaseFileHandler

logger = logging.getLogger(__name__)


class METimageL2NCFileHandler(METimageNCBaseFileHandler):
    """Reader class for VII L2 products in netCDF format."""

    def __init__(self, filename, filename_info, filetype_info, **kwargs):
        """Prepare the class for dataset reading."""
        orthorect = kwargs.pop("orthorect", False)
        super().__init__(filename, filename_info, filetype_info, orthorect=orthorect, **kwargs)

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
            # in the L2 case, the orthorectification correction data is already in degrees and can be applied directly
            variable += orthorect_data
        except KeyError:
            logger.warning("Required dataset %s for orthorectification not available, skipping", orthorect_data_name)
        return variable
