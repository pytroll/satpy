#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2019 Satpy developers
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
"""Composite classes for the LI instrument."""

import logging
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj as pro
import shapely
import xarray as xr

from satpy import config
from satpy.composites import CompositeBase

LOG = logging.getLogger(__name__)


class LightningTimeCompositor(CompositeBase):
      """Compositor class for lightning visualisation based on time.

      The compositor normalises the lightning event times between 0 and 1.
      The value 1 corresponds to the latest lightning event and the value 0 corresponds
      to the latest lightning event - time_range. The time_range is defined in the composite recipe
      and is in minutes.
      """
      def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
          """Initialisation of the class."""
          super().__init__(name, prerequisites, optional_prerequisites, **kwargs)
          # Get the time_range which is in minute
          self.time_range = self.attrs["time_range"]
          self.standard_name = self.attrs["standard_name"]
          self.reference_time_attr = self.attrs["reference_time"]


      def _normalize_time(self, data:xr.DataArray, attrs:dict) -> xr.DataArray:
          """Normalize the time in the range between [end_time, end_time - time_range].

          The range of the normalised data is between 0 and 1 where 0 corresponds to the date end_time - time_range
          and 1 to the end_time. Where end_times represent the latest lightning event and time_range is the range of
          time in minutes visualised in the composite.
          The dates that are earlier to end_time - time_range are set to NaN.

          Args:
              data (xr.DataArray): datas containing dates to be normalised
              attrs (dict): Attributes suited to the flash_age composite

          Returns:
              xr.DataArray: Normalised time
          """
          # Compute the maximum time value
          end_time = np.array(np.datetime64(data.attrs[self.reference_time_attr]))
          # Compute the minimum time value based on the time range
          begin_time = end_time - np.timedelta64(self.time_range, "m")
          # Invalidate values that are before begin_time
          condition_time = data >= begin_time
          data = data.where(condition_time)

          # raise a warning if data is empty after filtering
          if np.all(np.isnan(data)) :
              LOG.warning(f"All the flash_age events happened before {begin_time}, the composite will be empty.")

          # Normalize the time values
          normalized_data = (data - begin_time) / (end_time - begin_time)
          # Ensure the result is still an xarray.DataArray
          return xr.DataArray(normalized_data, dims=data.dims, coords=data.coords, attrs=attrs)


      @staticmethod
      def _update_missing_metadata(existing_attrs, new_attrs):
          for key, val in new_attrs.items():
              if key not in existing_attrs and val is not None:
                  existing_attrs[key] = val

      def _redefine_metadata(self,attrs:dict)->dict:
          """Modify the standard_name and name metadatas.

          Args:
              attrs (dict): data's attributes

          Returns:
              dict: updated attributes
          """
          attrs["name"] = self.standard_name
          attrs["standard_name"] = self.standard_name
          return attrs


      def __call__(self, projectables, nonprojectables=None, **attrs):
          """Normalise the dates."""
          data = projectables[0]
          new_attrs = data.attrs.copy()
          self._update_missing_metadata(new_attrs, attrs)
          new_attrs = self._redefine_metadata(new_attrs)
          return self._normalize_time(data, new_attrs)



class GeometryContainer:
    """Container for geometries stored in geopandas Geodataframes."""
    def __init__(self, data: gpd.GeoDataFrame, attrs=None):
        """Create Container for geodataframe."""
        if not isinstance(data, gpd.GeoDataFrame):
            raise TypeError("Data must be a GeoDataFrame")
        self.data = data
        self.attrs = attrs if attrs is not None else {}

    def __getitem__(self, key):
        """Get column by name."""
        # Basic slicing or column access
        return self.data[key]

    def __repr__(self):
        """Print geodataframe."""
        return f"<GeoDataFrame>\nData:\n{self.data}\n\nAttributes:\n{self.attrs}"

class FlashGeometry(CompositeBase):
    """Flash Geometry Processor."""

    def __init__(self, name, prerequisites=None, optional_prerequisites=None, **kwargs):
        """Initialisation of the class."""
        super().__init__(name, prerequisites, optional_prerequisites, **kwargs)
        self.standard_name = "acc_flash_geometry"

    def __call__(
            self,
            datasets: Sequence[xr.DataArray],
            optional_datasets: Optional[Sequence[xr.DataArray]] = None,
            **attrs
            ) -> gpd.GeoDataFrame:
        """Generate Flash Geometries."""
        distance_threshold = config.get("composites.flash_geom_distance_treshold", 10)

        ds = xr.Dataset(dict(zip(["flash_id", "group_time", "longitude", "latitude"], datasets))).compute()
        mintime = np.min(ds["group_time"])
        maxtime = np.max(ds["group_time"])
        duration = (maxtime - mintime)
        ds["normalized_group_time"] = (ds["group_time"] - mintime)/duration

        tdf = ds.to_dataframe()
        gb = tdf.groupby("flash_id")
        geom_df = gb.apply(lambda x: CreateFlashGeometry(x["longitude"].values, x["latitude"].values,
            x["normalized_group_time"].values[-1], mintime.values, distance_threshold))

        res = gpd.GeoDataFrame(geom_df, geometry="geometry").reset_index().drop(columns=["level_1"])
        res = res.set_crs(str(ds.crs.values))
        res = res.dissolve(by="flash_id", as_index=False)
        # remove empty geometries
        res = res[~res.geometry.is_empty].reset_index(drop=True)

        new_attrs={}
        new_attrs["name"] = "acc_flash_geometry"
        return GeometryContainer(data=res, attrs=new_attrs)

def CreateFlashGeometry(x, y, flashtime, group_time, distance_threshold):
    """Create flash geometry based on group centroids.

    Based on code [1] from Pieter Groenemeijer from ESSL create flash geometries.

    Args:
        x (list): lons of flash groups
        y (list): lats of flash groups
        flashtime (list): flashtime from li dataset
        group_time (list): group time from li dataset
        distance_threshold (int): Threshold value km below which connections between groups are made.
                                  This filters some unplausible long connections between groups. Defaults to 10km.

    Returns:
        pandas.DataFrame

    References:
        [1] https://gist.github.com/emiliom/87a6aa137610bf59787868943b406e8f
    """
    n = len(x)

    x1 = np.array(x)
    x1 = x1[:, np.newaxis]

    x2 = np.array(x)
    x2 = x2[:, np.newaxis].T

    y1 = np.array(y)
    y1 = y1[:, np.newaxis]

    y2 = np.array(y)
    y2 = y2[:, np.newaxis].T

    distances = np.sqrt((x2 - x1)**2 + 1.4 * (y2 - y1)**2)
    ind = list(range(0, n))
    distances[ind, ind] = np.nan

    connected = np.arange(0, n) * 0

    start = 0
    counter = 0
    connections = []

    geoms = []

    while counter < n:

        counter = counter + 1
        connected[start] = connected[start] + 1
        searchdistances = distances.copy()
        searchdistances[connected == 0, :] = np.nan
        searchdistances[:, connected > 0] = np.nan

        try:
            connection = np.nanargmin(searchdistances)
        except: # I.e. an array of NaNs results  # noqa E722
            break

        start = connection // searchdistances.shape[1]
        end = connection % searchdistances.shape[1]

        group_distance = dist(x[start], y[start], x[end], y[end])

        if group_distance <= distance_threshold:
            geoms.append(shapely.LineString([[x[start], y[start]], [x[end], y[end]]]))
            connected[end] = connected[end] + 1
            connections.append(connection)
        else:
            continue

    return pd.DataFrame({"geometry": geoms, "group_end_time": group_time, "normalized_group_time": flashtime})


def dist(lon1, lat1, lon2, lat2):  # noqa D417
    """Calculate distance between two points using pyproj.

    Args:
        lon1, lat1, lon2, lat2 (float): Coordinates in degrees.

    Returns:
        Distance in in km.
    """
    geod = pro.Geod(ellps="WGS84")
    _, _, distance = geod.inv(lon1, lat1, lon2, lat2)
    return distance / 1000
