#
"""This modules describes the sarx instrument from the TerraSAR-X satellite.
"""
import numpy as np

from mpop.compositer import Compositer
from mpop.logger import LOG

class SarxCompositer(Compositer):
    """This class sets up the SAR-X instrument channel list.
    """
    
    instrument_name = "sarx"

    def average(self, downscaling_factor=2, average_window=None):
        from mpop.imageo.geo_image import GeoImage
        from pyresample import geometry
        import scipy.ndimage as ndi

        self.check_channels(9.65)

        if average_window == None:
            average_window = downscaling_factor

        LOG.info("Downsampling a factor %d and averaging "%downscaling_factor + 
                 "in a window of %dx%d"%(average_window, average_window))

        ch = self[9.65]

        # If average window and downscale factor is the same
        # the following could be used:
        #
        #    data = data.reshape([shight, hight/shight,
        #                         swidth, width/swidth]).mean(3).mean(1)

        # avg kernel
        kernel = np.ones((average_window, average_window), dtype=np.float) / \
            (average_window*average_window)
        # do convolution
        data = ndi.filters.correlate(ch.data.astype(np.float), kernel, mode='nearest')
        # downscale
        data = data[1::downscaling_factor, 1::downscaling_factor]

        # New area
        area = geometry.AreaDefinition(ch.area.area_id, ch.area.name,
                                       ch.area.proj_id, ch.area.proj_dict,
                                       data.shape[1], data.shape[0],
                                       ch.area.area_extent)

        img = GeoImage(data, area, self.time_slot,
                       fill_value=(0,), mode='L')
        return img

    average.prerequisites = set([9.65,])
    
