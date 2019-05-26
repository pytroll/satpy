from trollimage.colormap import Colormap
import numpy as np
from satpy.enhancements import apply_enhancement


def water_detection(img, **kwargs):
    """Palettizes images from VIIRS flood data.

    This modifies the image's data so the correct colors
    can be applied to it, and then palettizes the image.
    """
    palette = kwargs['palettes']
    palette['colors'] = tuple(map(tuple, palette['colors']))

    def func(img_data):
        data = np.asarray(img_data)
        data[data == 150] = 31
        data[data == 199] = 18
        data[data >= 200] = data[data >= 200] - 100

        return data

    apply_enhancement(img.data, func, pass_dask=True)
    cm = Colormap(*palette['colors'])
    img.palettize(cm)
