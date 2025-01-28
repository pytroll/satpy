# Copyright (c) 2024 - 2025 Pytroll Developers

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""Reader for the Arctic Weather Satellite (AWS) MWR level-1c data.

MWR = Microwave Radiometer, onboard AWS and EPS-Sterna

Sample data provided by ESA September 27, 2024.


Example:
--------
Here is an example how to read the data in satpy:

.. code-block:: python

    from satpy import Scene
    from glob import glob

    filenames = glob("data/W_XX-OHB-Stockholm,SAT,AWS1-MWR-1C-RAD_C_OHB_*20240913204851_*.nc")

    scn = Scene(filenames=filenames, reader='aws1_mwr_l1c_nc')

    composites = ['mw183_humidity']
    dataset_names = composites + ['1']

    scn.load(dataset_names)
    print(scn['1'])
    scn.show('mw183_humidity')

"""


from satpy.readers.mwr_l1b import MWR_CHANNEL_NAMES, AWS_EPS_Sterna_BaseFileHandler, mask_and_scale


class AWS_MWR_L1CFile(AWS_EPS_Sterna_BaseFileHandler):
    """Class implementing the AWS L1c Filehandler.

    This class implements the ESA Arctic Weather Satellite (AWS) Level-1b
    NetCDF reader. It is designed to be used through the :class:`~satpy.Scene`
    class using the :mod:`~satpy.Scene.load` method with the reader
    ``"aws_l1c_nc"``.

    """
    def __init__(self, filename, filename_info, filetype_info, auto_maskandscale=True):
        """Initialize the handler."""
        super().__init__(filename, filename_info, filetype_info, auto_maskandscale)
        self.filename_info = filename_info

    @property
    def sensor(self):
        """Get the sensor name."""
        # This should have been self["/attr/instrument"]
        # But the sensor name is currently incorrect in the ESA level-1b files
        return "mwr"

    def get_dataset(self, dataset_id, dataset_info):
        """Get the data."""
        if dataset_id["name"] in MWR_CHANNEL_NAMES:
            data_array = self._get_channel_data(dataset_id, dataset_info)
        elif (dataset_id["name"] in ["longitude", "latitude",
                                     "solar_azimuth_angle", "solar_zenith_angle",
                                     "satellite_zenith_angle", "satellite_azimuth_angle"]):
            data_array = self._get_navigation_data(dataset_id, dataset_info)
        else:
            raise NotImplementedError(f"Dataset {dataset_id['name']} not available or not supported yet!")

        data_array = mask_and_scale(data_array)
        if dataset_id["name"] == "longitude":
            data_array = data_array.where(data_array <= 180, data_array - 360)

        data_array.attrs.update(dataset_info)

        data_array.attrs["platform_name"] = self.platform_name
        data_array.attrs["sensor"] = self.sensor
        return data_array


    def _get_navigation_data(self, dataset_id, dataset_info):
        """Get the navigation (geolocation) data."""
        geo_data = self[dataset_info["file_key"]]
        geo_data = geo_data.rename({"n_fovs": "x", "n_scans": "y"})
        return geo_data
