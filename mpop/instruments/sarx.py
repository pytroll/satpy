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

    def average(self, downscaling_factor=2, average_window=None,
                stretch="linear", clip_percentage=None):
        """
        Makes a mean convolution of an image.

        :Parameters:
         `downscaling_factor` : int
             image downscaling factor, default is a factor 2.             
         `average_window` : int
             window size for calculating mean values, default is
             the same as downscaling_factor.
         `stretch` : string
             enhance methode, default is `linear`.
         `clip_percentage` : tuple
             percentage discard in near and far end range.

        :Returns:
         `image` : GeoImage
             mean convoluted image.
        """
        from mpop.imageo.geo_image import GeoImage
        from pyresample import geometry
        import scipy.ndimage as ndi

        self.check_channels(9.65)

        if average_window == None:
            average_window = downscaling_factor

        ch = self[9.65]

        LOG.info("Downsampling a factor %d and averaging "%downscaling_factor + 
                 "in a window of %dx%d"%(average_window, average_window))
        
        # If average window and downscale factor is the same
        # the following could be used:
        #
        #    data = data.reshape([shight, hight/shight,
        #                         swidth, width/swidth]).mean(3).mean(1)

        # avg kernel
        data = ch.data.astype(np.float).filled(np.nan)
        kernel = np.ones((average_window, average_window), dtype=np.float) / \
            (average_window*average_window)
        # do convolution
        data = ndi.filters.correlate(data, kernel, mode='nearest')
        data = np.ma.numpy(data, mask=
        # downscale
        data = data[1::downscaling_factor, 1::downscaling_factor]

        # New area, and correct for integer truncation.
        p_size_x, p_size_y = (ch.area.pixel_size_x*downscaling_factor, 
                              ch.area.pixel_size_y*downscaling_factor)
        area_extent = (ch.area.area_extent[0],
                       ch.area.area_extent[1],
                       ch.area.area_extent[0] + data.shape[1]*p_size_x,
                       ch.area.area_extent[1] + data.shape[0]*p_size_y)
                       
        area = geometry.AreaDefinition(self._data_holder.satname + 
                                       self._data_holder.instrument_name +
                                       str(area_extent) +
                                       str(data.shape),
                                       "On-the-fly area",
                                       ch.area.proj_id, ch.area.proj_dict,
                                       data.shape[1], data.shape[0],
                                       area_extent)

        if clip_percentage != None:
            from scipy import stats
            cdata = data.flatten() #.compressed()
            cdata = cdata[cdata != np.nan]
            score = (stats.scoreatpercentile(cdata, clip_percentage[0]),
                     stats.scoreatpercentile(cdata, 100. - clip_percentage[1]))
            LOG.info("Clipping data %f%%, %f%% (%f, %f) in far and near edge", 
                     clip_percentage[0], clip_percentage[1], score[0], score[1])
            data = np.clip(data, score[0], score[1], data)
        
        img = GeoImage(data, area, self.time_slot,
                       fill_value=(0,), mode='L')
        if stretch:
            img.enhance(stretch=stretch)
        return img

    average.prerequisites = set([9.65,])
    
