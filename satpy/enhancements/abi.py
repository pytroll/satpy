"""Enhancement functions specific to the ABI sensor."""

from satpy.enhancements.wrappers import exclude_alpha, using_map_blocks


def cimss_true_color_contrast(img, **kwargs):
    """Scale data based on CIMSS True Color recipe for AWIPS."""
    _cimss_true_color_contrast(img.data)


@exclude_alpha
@using_map_blocks
def _cimss_true_color_contrast(img_data):
    """Perform per-chunk enhancement.

    Code ported from Kaba Bah's AWIPS python plugin for creating the
    CIMSS Natural (True) Color image in AWIPS. AWIPS provides that python
    code the image data on a 0-255 scale. Satpy gives this function the
    data on a 0-1.0 scale (assuming linear stretching and sqrt
    enhancements have already been applied).

    """
    max_value = 1.0
    acont = (255.0 / 10.0) / 255.0
    amax = (255.0 + 4.0) / 255.0
    amid = 1.0 / 2.0
    afact = (amax * (acont + max_value) / (max_value * (amax - acont)))
    aband = (afact * (img_data - amid) + amid)
    aband[aband <= 10 / 255.0] = 0
    aband[aband >= 1.0] = 1.0
    return aband
