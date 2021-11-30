import warnings

import numpy as np
import pyproj
from pyresample.geometry import AreaDefinition
from scipy.ndimage import map_coordinates


def parse_area(area_def):
    if isinstance(area_def, AreaDefinition):
        area = area_def
    elif isinstance(area_def, dict):
        with warnings.catch_warnings(): # suppress warnings
            warnings.simplefilter("ignore", category=UserWarning)
            try:
                area = AreaDefinition(**area_def)
            except TypeError:
                area_def = known_area(area_def)
                area = AreaDefinition(**area_def)
    else:
        print("*** Error, unknown area_def")
                
                
    return area


def known_area(area_dict):
    assert("type" in area_dict)
    if area_dict["type"] == "swiss_ccs4":
        return ccs4_swiss_grid_area
    elif area_dict["type"] == "azimuthal_equidistant":
        return centered_aeqd_area(**area_dict["params"])


class GridProjection:
    def __init__(self, area_def):
        self.area = parse_area(area_def)

        self.grid_proj = pyproj.Proj(**self.area.proj_dict)
        self.latlon_proj = pyproj.Proj(proj='latlong')
        self.transformer = pyproj.Transformer.from_proj(
            self.latlon_proj, self.grid_proj)

    def __call__(self, lon, lat):
        # the pyproj call below does not handle empty inputs properly
        if (not np.isscalar(lon)) and (len(lon) == 0) \
            and (not np.isscalar(lat)) and (len(lat) == 0):
            return (np.copy(lon), np.copy(lat))

        # get cartesian projection values from longitude and latitude
        (source_x, source_y) = self.transformer.transform(lon, lat)

        # Find corresponding pixels (element by element conversion of ndarrays)
        # borrowed from:
        # https://github.com/pytroll/pyresample/blob/master/pyresample/grid.py
        source_pixel_x = (self.area.pixel_offset_x +
            source_x / self.area.pixel_size_x)

        source_pixel_y = (self.area.pixel_offset_y -
            source_y / self.area.pixel_size_y)

        return (source_pixel_y, source_pixel_x)

    def inverse(self, y, x):
        # the pyproj call below does not handle empty inputs properly
        if (not np.isscalar(y)) and (len(x) == 0) \
            and (not np.isscalar(y)) and (len(x) == 0):
            return (np.copy(y), np.copy(x))

        proj_x = (x - self.area.pixel_offset_x) * self.area.pixel_size_x
        proj_y = (self.area.pixel_offset_y - y) * self.area.pixel_size_y

        # get longitude and latitude from cartesian projection values
        return self.transformer.transform(proj_x, proj_y,
            direction='inverse')


# Adapted from: 
# https://github.com/meteoswiss-mdr/monti-pytroll/blob/master/etc/areas.def
ccs4_swiss_grid_area = {
    "area_id": "ccs4",
    "description": "CCS4 Swiss Grid",
    "proj_id": "epsg:21781",
    "projection": "epsg:21781",
    "width": 710,
    "height": 640,
    "area_extent": (255000.0, -160000.0, 965000.0, 480000.0)
}

def geostationary_area(
        *, area_id, description, a, b, lon_0, h,
        nrow, ncol, coff, cfac, loff, lfac  
    ):

    # from mpop/satin/nwcsaf_msg.py
    xur = (ncol - coff) * 2 ** 16 / (cfac * 1.0) 
    xur = np.deg2rad(xur) * h
    xll = (-1 - coff) * 2 ** 16 / (cfac * 1.0)
    xll = np.deg2rad(xll) * h
    xres = (xur - xll) / ncol
    (xur, xll) = (xur - xres / 2, xll + xres / 2)
    yll = (nrow - loff) * 2 ** 16 / (-lfac * 1.0)
    yll = np.deg2rad(yll) * h
    yur = (-1 - loff) * 2 ** 16 / (-lfac * 1.0)
    yur = np.deg2rad(yur) * h
    yres = (yur - yll) / nrow
    (yll, yur) = (yll + yres / 2, yur - yres / 2)
    area_extent = (xll, yll, xur, yur)

    area = {
        "area_id": area_id,
        "description": description,
        "proj_id": "geos",
        "projection": {
            "proj": "geos",
            "a": a,
            "b": b,
            "lon_0": lon_0,
            "h": h
        },
        "height": nrow,
        "width": ncol,
        "area_extent": area_extent
    }
    return AreaDefinition(**area)

def geostationary_area_alps():
    return geostationary_area(
        area_id="alps",
        description="Alps Region Geostationary Projection",
        a=6378169.0, b=6356583.8,
        lon_0=9.5, h=35785831.0,
        nrow=151, ncol=349,
        coff=204, cfac=13642337,
        loff=1515, lfac=13642337        
    )

def centered_aeqd_area(lon0, lat0, size, pixel_size=1000.0):
    """Azimuthal equidistant projection centered on a given lon/lat point.
    """

    physical_size = (size[0]*pixel_size, size[1]*pixel_size)

    area = {
        "area_id": "aeqd:{:.3f}:{:.3f}:{}km:{}x{}".format(
            lon0, lat0, pixel_size, size[0], size[1]
        ),
        "description": "Azimuthal Equidistant Projection " + \
            "lon0={:.3f}, lat0={:.3f}, scale={}km, {}x{}".format(
                lon0, lat0, pixel_size, size[0], size[1]
        ),
        "proj_id": "aeqd:{:.3f}:{:.3f}".format(lon0,lat0),
        "projection": {
            "proj": "aeqd",
            "lon_0": lon0,
            "lat_0": lat0,
        },
        "width": size[0],
        "height": size[1],
        "area_extent": (
            -physical_size[0]/2, 
            -physical_size[1]/2,
            physical_size[0]/2, 
            physical_size[1]/2
        )
    }

    return area


class ImageMapper:
    def __init__(self, source_area, target_area):
        self.source_area = source_area
        self.target_area = target_area
        source_proj = GridProjection(source_area)
        target_proj = GridProjection(target_area)
        (y_src, x_src) = np.mgrid[
            :target_area.height,
            :target_area.width
        ]
        (self.y_tgt, self.x_tgt) = source_proj(*target_proj.inverse(y_src, x_src))
       
    def __call__(self, image, **kwargs):
        return map_coordinates(image, (self.y_tgt, self.x_tgt), **kwargs)
