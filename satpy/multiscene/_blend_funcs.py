from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd
import xarray as xr
from dask import array as da

from satpy.dataset import combine_metadata


def stack(
        data_arrays: Sequence[xr.DataArray],
        weights: Optional[Sequence[xr.DataArray]] = None,
        combine_times: bool = True,
        blend_type: str = 'select_with_weights'
) -> xr.DataArray:
    """Combine a series of datasets in different ways.

    By default, DataArrays are stacked on top of each other, so the last one
    applied is on top. Each DataArray is assumed to represent the same
    geographic region, meaning they have the same area. If a sequence of
    weights is provided then they must have the same shape as the area.
    Weights with greater than 2 dimensions are not currently supported.

    When weights are provided, the DataArrays will be combined according to
    those weights. Data can be integer category products (ex. cloud type),
    single channels (ex. radiance), or a multi-band composite (ex. an RGB or
    RGBA true_color). In the latter case, the weight array is applied
    to each band (R, G, B, A) in the same way. The result will be a composite
    DataArray where each pixel is constructed in a way depending on ``blend_type``.

    Blend type can be one of the following:

     * select_with_weights: The input pixel with the maximum weight is chosen.
     * blend_with_weights: The final pixel is a weighted average of all valid
       input pixels.

    """
    if weights:
        return _stack_with_weights(data_arrays, weights, combine_times, blend_type)
    return _stack_no_weights(data_arrays, combine_times)


def _stack_with_weights(
        datasets: Sequence[xr.DataArray],
        weights: Sequence[xr.DataArray],
        combine_times: bool,
        blend_type: str
) -> xr.DataArray:
    blend_func = _get_weighted_blending_func(blend_type)
    filled_weights = list(_fill_weights_for_invalid_dataset_pixels(datasets, weights))
    return blend_func(datasets, filled_weights, combine_times)


def _get_weighted_blending_func(blend_type: str) -> Callable:
    WEIGHTED_BLENDING_FUNCS = {
        "select_with_weights": _stack_select_by_weights,
        "blend_with_weights": _stack_blend_by_weights,
    }
    blend_func = WEIGHTED_BLENDING_FUNCS.get(blend_type)
    if blend_func is None:
        raise ValueError(f"Unknown weighted blending type: {blend_type}."
                         f"Expected one of: {WEIGHTED_BLENDING_FUNCS.keys()}")
    return blend_func


def _fill_weights_for_invalid_dataset_pixels(
        datasets: Sequence[xr.DataArray],
        weights: Sequence[xr.DataArray]
) -> Iterable[xr.DataArray]:
    """Replace weight valus with 0 where data values are invalid/null."""
    has_bands_dims = "bands" in datasets[0].dims
    for i, dataset in enumerate(datasets):
        # if multi-band only use the red-band
        compare_ds = dataset[0] if has_bands_dims else dataset
        try:
            yield xr.where(compare_ds == compare_ds.attrs["_FillValue"], 0, weights[i])
        except KeyError:
            yield xr.where(compare_ds.isnull(), 0, weights[i])


def _stack_blend_by_weights(
        datasets: Sequence[xr.DataArray],
        weights: Sequence[xr.DataArray],
        combine_times: bool
) -> xr.DataArray:
    """Stack datasets blending overlap using weights."""
    attrs = _combine_stacked_attrs([data_arr.attrs for data_arr in datasets], combine_times)

    overlays = []
    for weight, overlay in zip(weights, datasets):
        # Any 'overlay' fill values should already be reflected in the weights
        # as 0. See _fill_weights_for_invalid_dataset_pixels. We fill NA with
        # 0 here to avoid NaNs affecting valid pixels in other datasets. Note
        # `.fillna` does not handle the `_FillValue` attribute so this filling
        # is purely to remove NaNs.
        overlays.append(overlay.fillna(0) * weight)
    # NOTE: Currently no way to ignore numpy divide by 0 warnings without
    # making a custom map_blocks version of the divide
    base = sum(overlays) / sum(weights)

    dims = datasets[0].dims
    blended_array = xr.DataArray(base, dims=dims, attrs=attrs)
    return blended_array


def _stack_select_by_weights(
        datasets: Sequence[xr.DataArray],
        weights: Sequence[xr.DataArray],
        combine_times: bool
) -> xr.DataArray:
    """Stack datasets selecting pixels using weights."""
    indices = da.argmax(da.dstack(weights), axis=-1)
    if "bands" in datasets[0].dims:
        indices = [indices] * datasets[0].sizes["bands"]

    attrs = _combine_stacked_attrs([data_arr.attrs for data_arr in datasets], combine_times)
    dims = datasets[0].dims
    coords = datasets[0].coords
    selected_array = xr.DataArray(da.choose(indices, datasets), dims=dims, coords=coords, attrs=attrs)
    return selected_array


def _stack_no_weights(
        datasets: Sequence[xr.DataArray],
        combine_times: bool
) -> xr.DataArray:
    base = datasets[0].copy()
    collected_attrs = [base.attrs]
    for data_arr in datasets[1:]:
        collected_attrs.append(data_arr.attrs)
        try:
            base = base.where(data_arr == data_arr.attrs["_FillValue"], data_arr)
        except KeyError:
            base = base.where(data_arr.isnull(), data_arr)

    attrs = _combine_stacked_attrs(collected_attrs, combine_times)
    base.attrs = attrs
    return base


def _combine_stacked_attrs(collected_attrs: Sequence[Mapping], combine_times: bool) -> dict:
    attrs = combine_metadata(*collected_attrs)
    if combine_times and ('start_time' in attrs or 'end_time' in attrs):
        new_start, new_end = _get_combined_start_end_times(collected_attrs)
        if new_start:
            attrs["start_time"] = new_start
        if new_end:
            attrs["end_time"] = new_end
    return attrs


def _get_combined_start_end_times(metadata_objects: Iterable[Mapping]) -> tuple[datetime | None, datetime | None]:
    """Get the start and end times attributes valid for the entire dataset series."""
    start_time = None
    end_time = None
    for md_obj in metadata_objects:
        if "start_time" in md_obj and (start_time is None or md_obj['start_time'] < start_time):
            start_time = md_obj['start_time']
        if "end_time" in md_obj and (end_time is None or md_obj['end_time'] > end_time):
            end_time = md_obj['end_time']
    return start_time, end_time


def timeseries(datasets):
    """Expand dataset with and concatenate by time dimension."""
    expanded_ds = []
    for ds in datasets:
        if 'time' not in ds.dims:
            tmp = ds.expand_dims("time")
            tmp.coords["time"] = pd.DatetimeIndex([ds.attrs["start_time"]])
        else:
            tmp = ds
        expanded_ds.append(tmp)

    res = xr.concat(expanded_ds, dim="time")
    res.attrs = combine_metadata(*[x.attrs for x in expanded_ds])
    return res


def temporal_rgb(
        data_arrays: Sequence[xr.DataArray],
) -> xr.DataArray:
    """Combine a series of datasets as a temporal RGB.

    The first dataset is used as the Red component of the new composite, the second as Green and the third as Blue.
    All the other datasets are discarded.
    """
    from satpy.composites import GenericCompositor

    compositor = GenericCompositor("temporal_composite")
    composite = compositor((data_arrays[0], data_arrays[1], data_arrays[2]))
    composite.attrs = data_arrays[2].attrs

    return composite
