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

import numpy as np
import xarray as xr

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
