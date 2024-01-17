# Copyright (c) 2023 Satpy developers
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
"""Helper functions for converting the Scene object to some other object."""

import xarray as xr

from satpy.composites import enhance2dataset
from satpy.dataset import DataID


def _get_dataarrays_from_identifiers(scn, identifiers):
    """Return a list of DataArray based on a single or list of identifiers.

    An identifier can be a DataID or a string with name of a valid DataID.
    """
    if isinstance(identifiers, (str, DataID)):
        identifiers = [identifiers]

    if identifiers is not None:
        dataarrays = [scn[ds] for ds in identifiers]
    else:
        dataarrays = [scn._datasets.get(ds) for ds in scn._wishlist]
        dataarrays = [dataarray for dataarray in dataarrays if dataarray is not None]
    return dataarrays


def to_geoviews(scn, gvtype=None, datasets=None,
                kdims=None, vdims=None, dynamic=False):
        """Convert satpy Scene to geoviews.

        Args:
            scn (satpy.Scene): Satpy Scene.
            gvtype (gv plot type):
                One of gv.Image, gv.LineContours, gv.FilledContours, gv.Points
                Default to :class:`geoviews.Image`.
                See Geoviews documentation for details.
            datasets (list): Limit included products to these datasets
            kdims (list of str):
                Key dimensions. See geoviews documentation for more information.
            vdims (list of str, optional):
                Value dimensions. See geoviews documentation for more information.
                If not given defaults to first data variable
            dynamic (bool, optional): Load and compute data on-the-fly during
                visualization. Default is ``False``. See
                https://holoviews.org/user_guide/Gridded_Datasets.html#working-with-xarray-data-types
                for more information. Has no effect when data to be visualized
                only has 2 dimensions (y/x or longitude/latitude) and doesn't
                require grouping via the Holoviews ``groupby`` function.

        Returns: geoviews object

        Todo:
            * better handling of projection information in datasets which are
              to be passed to geoviews

        """
        import geoviews as gv
        from cartopy import crs  # noqa
        if gvtype is None:
            gvtype = gv.Image

        ds = scn.to_xarray_dataset(datasets)

        if vdims is None:
            # by default select first data variable as display variable
            vdims = ds.data_vars[list(ds.data_vars.keys())[0]].name

        if hasattr(ds, "area") and hasattr(ds.area, "to_cartopy_crs"):
            dscrs = ds.area.to_cartopy_crs()
            gvds = gv.Dataset(ds, crs=dscrs)
        else:
            gvds = gv.Dataset(ds)

        # holoviews produces a log warning if you pass groupby arguments when groupby isn't used
        groupby_kwargs = {"dynamic": dynamic} if gvds.ndims != 2 else {}
        if "latitude" in ds.coords:
            gview = gvds.to(gv.QuadMesh, kdims=["longitude", "latitude"],
                            vdims=vdims, **groupby_kwargs)
        else:
            gview = gvds.to(gvtype, kdims=["x", "y"], vdims=vdims,
                            **groupby_kwargs)

        return gview

def to_hvplot(scn, datasets=None, *args, **kwargs):
        """Convert satpy Scene to Hvplot. The method could not be used with composites of swath data.

        Args:
            scn (satpy.Scene): Satpy Scene.
            datasets (list): Limit included products to these datasets.
            args: Arguments coming from hvplot
            kwargs: hvplot options dictionary.

        Returns:
            hvplot object that contains within it the plots of datasets list.
            As default it contains all Scene datasets plots and a plot title
            is shown.

        Example usage::

           scene_list = ['ash','IR_108']
           scn = Scene()
           scn.load(scene_list)
           scn = scn.resample('eurol')
           plot = scn.to_hvplot(datasets=scene_list)
           plot.ash+plot.IR_108
        """

        def _get_crs(xarray_ds):
            return xarray_ds.area.to_cartopy_crs()

        def _get_timestamp(xarray_ds):
            time = xarray_ds.attrs["start_time"]
            return time.strftime("%Y %m %d -- %H:%M UTC")

        def _get_units(xarray_ds, variable):
            return xarray_ds[variable].attrs["units"]

        def _plot_rgb(xarray_ds, variable, **defaults):
            img = enhance2dataset(xarray_ds[variable])
            return img.hvplot.rgb(bands="bands", title=title,
                                  clabel="", **defaults)

        def _plot_quadmesh(xarray_ds, variable, **defaults):
            return xarray_ds[variable].hvplot.quadmesh(
                clabel=f"[{_get_units(xarray_ds,variable)}]", title=title,
                **defaults)

        import hvplot.xarray as hvplot_xarray  # noqa
        from holoviews import Overlay

        plot = Overlay()
        xarray_ds = scn.to_xarray_dataset(datasets)

        if hasattr(xarray_ds, "area") and hasattr(xarray_ds.area, "to_cartopy_crs"):
            ccrs = _get_crs(xarray_ds)
            defaults={"x":"x","y":"y"}
        else:
            ccrs = None
            defaults={"x":"longitude","y":"latitude"}

        if datasets is None:
            datasets = list(xarray_ds.keys())

        defaults.update(data_aspect=1, project=True, geo=True,
                        crs=ccrs, projection=ccrs, rasterize=True,
                        coastline="110m", cmap="Plasma", responsive=True,
                        dynamic=False, framewise=True,colorbar=False,
                        global_extent=False, xlabel="Longitude",
                        ylabel="Latitude")

        defaults.update(kwargs)

        for element in datasets:
            title = f"{element} @ {_get_timestamp(xarray_ds)}"
            if xarray_ds[element].shape[0] == 3:
                plot[element] = _plot_rgb(xarray_ds, element, **defaults)
            else:
                plot[element] = _plot_quadmesh(xarray_ds, element, **defaults)

        return plot

def to_xarray(scn,
              datasets=None,  # DataID
              header_attrs=None,
              exclude_attrs=None,
              flatten_attrs=False,
              pretty=True,
              include_lonlats=True,
              epoch=None,
              include_orig_name=True,
              numeric_name_prefix="CHANNEL_"):
    """Merge all xr.DataArray(s) of a satpy.Scene to a CF-compliant xarray object.

    If all Scene DataArrays are on the same area, it returns an xr.Dataset.
    If Scene DataArrays are on different areas, currently it fails, although
    in future we might return a DataTree object, grouped by area.

    Args:
        scn (satpy.Scene): Satpy Scene.
        datasets (iterable, optional): List of Satpy Scene datasets to include in
            the output xr.Dataset. Elements can be string name, a wavelength as a
            number, a DataID, or DataQuery object. If None (the default), it
            includes all loaded Scene datasets.
        header_attrs: Global attributes of the output xr.Dataset.
        epoch (str, optional): Reference time for encoding the time coordinates
            (if available). Format example: "seconds since 1970-01-01 00:00:00".
            If None, the default reference time is retrieved using
            "from satpy.cf_writer import EPOCH".
        flatten_attrs (bool, optional): If True, flatten dict-type attributes.
        exclude_attrs (list, optional): List of xr.DataArray attribute names to
            be excluded.
        include_lonlats (bool, optional): If True, includes 'latitude' and
            'longitude' coordinates. If the 'area' attribute is a SwathDefinition,
            it always includes latitude and longitude coordinates.
        pretty (bool, optional): Don't modify coordinate names, if possible. Makes
            the file prettier, but possibly less consistent.
        include_orig_name (bool, optional): Include the original dataset name as a
            variable attribute in the xr.Dataset.
        numeric_name_prefix (str, optional): Prefix to add to each variable with
            name starting with a digit. Use '' or None to leave this out.

    Returns:
        xr.Dataset: A CF-compliant xr.Dataset

    """
    from satpy.cf.datasets import collect_cf_datasets

    # Get list of DataArrays
    if datasets is None:
        datasets = list(scn.keys())  # list all loaded DataIDs
    list_dataarrays = _get_dataarrays_from_identifiers(scn, datasets)

    # Check that some DataArray could be returned
    if len(list_dataarrays) == 0:
        return xr.Dataset()

    # Collect xr.Dataset for each group
    grouped_datasets, header_attrs = collect_cf_datasets(list_dataarrays=list_dataarrays,
                                                         header_attrs=header_attrs,
                                                         exclude_attrs=exclude_attrs,
                                                         flatten_attrs=flatten_attrs,
                                                         pretty=pretty,
                                                         include_lonlats=include_lonlats,
                                                         epoch=epoch,
                                                         include_orig_name=include_orig_name,
                                                         numeric_name_prefix=numeric_name_prefix,
                                                         groups=None)
    if len(grouped_datasets) == 1:
        ds = grouped_datasets[None]
        return ds
    else:
        msg = """The Scene object contains datasets with different areas.
                  Resample the Scene to have matching dimensions using i.e. scn.resample(resampler="native") """
        raise NotImplementedError(msg)
