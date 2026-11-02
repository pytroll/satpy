"""Enhancements specific to the VIIRS instrument."""
import numpy as np
from trollimage.colormap import Colormap

from satpy.enhancements.wrappers import exclude_alpha, using_map_blocks


def water_detection(img, **kwargs):
    """Palettizes images from VIIRS flood data.

    This modifies the image's data so the correct colors
    can be applied to it, and then palettizes the image.
    """
    palette = kwargs["palettes"]
    palette["colors"] = tuple(map(tuple, palette["colors"]))

    _water_detection(img.data)
    cm = Colormap(*palette["colors"])
    img.palettize(cm)


@exclude_alpha
@using_map_blocks
def _water_detection(img_data):
    data = np.asarray(img_data).copy()
    data[data == 150] = 31
    data[data == 199] = 18
    data[data >= 200] = data[data >= 200] - 100

    return data
