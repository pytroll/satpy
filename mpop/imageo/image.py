#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009, 2012, 2013, 2014.

# SMHI,
# Folkborgsvägen 1,
# Norrköping,
# Sweden

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>
#   Esben S. Nielsen <esn@dmi.dk>

# This file is part of the mpop.

# mpop is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# mpop is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with mpop.  If not, see <http://www.gnu.org/licenses/>.

"""This module defines the image class. It overlaps largely the PIL library,
but has the advandage of using masked arrays as pixel arrays, so that data
arrays containing invalid values may be properly handled.
"""
import warnings
warnings.warn(__name__ + " is deprecated, please use trollimage instead.",
              DeprecationWarning)

import os
import re

from PIL import Image as Pil
import numpy as np

try:
    import numexpr as ne
except ImportError:
    ne = None

from mpop.utils import ensure_dir

import logging

LOG = logging.getLogger(__name__)


class UnknownImageFormat(Exception):

    """Exception to be raised when image format is unknown to MPOP"""
    pass


def check_image_format(fformat):
    cases = {"jpg": "jpeg",
             "jpeg": "jpeg",
             "tif": "tiff",
             "tiff": "tif",
             "pgm": "ppm",
             "pbm": "ppm",
             "ppm": "ppm",
             "bmp": "bmp",
             "dib": "bmp",
             "gif": "gif",
             "im": "im",
             "pcx": "pcx",
             "png": "png",
             "xbm": "xbm",
             "xpm": "xpm",
             }
    fformat = fformat.lower()
    try:
        fformat = cases[fformat]
    except KeyError:
        raise UnknownImageFormat("Unknown image format '%s'." % fformat)
    return fformat

try:
    from numpy import percentile
except ImportError:

    # Stolen from numpy 1.7.0 for backward compatibility

    def percentile(a, q, axis=None, out=None, overwrite_input=False):
        """
        Compute the qth percentile of the data along the specified axis.

        Returns the qth percentile of the array elements.

        Parameters
        ----------
        a : array_like
        Input array or object that can be converted to an array.
        q : float in range of [0,100] (or sequence of floats)
        Percentile to compute which must be between 0 and 100 inclusive.
        axis : int, optional
        Axis along which the percentiles are computed. The default (None)
        is to compute the median along a flattened version of the array.
        out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type (of the output) will be cast if necessary.
        overwrite_input : bool, optional
        If True, then allow use of memory of input array `a` for
        calculations. The input array will be modified by the call to
        median. This will save memory when you do not need to preserve
        the contents of the input array. Treat the input as undefined,
        but it will probably be fully or partially sorted.
        Default is False. Note that, if `overwrite_input` is True and the
        input is not already an array, an error will be raised.

        Returns
        -------
        pcntile : ndarray
        A new array holding the result (unless `out` is specified, in
        which case that array is returned instead). If the input contains
        integers, or floats of smaller precision than 64, then the output
        data-type is float64. Otherwise, the output data-type is the same
        as that of the input.

        See Also
        --------
        mean, median

        Notes
        -----
        Given a vector V of length N, the qth percentile of V is the qth ranked
        value in a sorted copy of V. A weighted average of the two nearest
        neighbors is used if the normalized ranking does not match q exactly.
        The same as the median if ``q=50``, the same as the minimum if ``q=0``
        and the same as the maximum if ``q=100``.

        Examples
        --------
        >>> a = np.array([[10, 7, 4], [3, 2, 1]])
        >>> a
        array([[10, 7, 4],
        [ 3, 2, 1]])
        >>> np.percentile(a, 50)
        3.5
        >>> np.percentile(a, 0.5, axis=0)
        array([ 6.5, 4.5, 2.5])
        >>> np.percentile(a, 50, axis=1)
        array([ 7., 2.])

        >>> m = np.percentile(a, 50, axis=0)
        >>> out = np.zeros_like(m)
        >>> np.percentile(a, 50, axis=0, out=m)
        array([ 6.5, 4.5, 2.5])
        >>> m
        array([ 6.5, 4.5, 2.5])

        >>> b = a.copy()
        >>> np.percentile(b, 50, axis=1, overwrite_input=True)
        array([ 7., 2.])
        >>> assert not np.all(a==b)
        >>> b = a.copy()
        >>> np.percentile(b, 50, axis=None, overwrite_input=True)
        3.5

        """
        a = np.asarray(a)

        if q == 0:
            return a.min(axis=axis, out=out)
        elif q == 100:
            return a.max(axis=axis, out=out)

        if overwrite_input:
            if axis is None:
                sorted = a.ravel()
                sorted.sort()
            else:
                a.sort(axis=axis)
                sorted = a
        else:
            sorted = np.sort(a, axis=axis)
        if axis is None:
            axis = 0

        return _compute_qth_percentile(sorted, q, axis, out)

    # handle sequence of q's without calling sort multiple times
    def _compute_qth_percentile(sorted, q, axis, out):
        if not np.isscalar(q):
            p = [_compute_qth_percentile(sorted, qi, axis, None)
                 for qi in q]

            if out is not None:
                out.flat = p

            return p

        q = q / 100.0
        if (q < 0) or (q > 1):
            raise ValueError("percentile must be either in the range [0,100]")

        indexer = [slice(None)] * sorted.ndim
        Nx = sorted.shape[axis]
        index = q * (Nx - 1)
        i = int(index)
        if i == index:
            indexer[axis] = slice(i, i + 1)
            weights = np.array(1)
            sumval = 1.0
        else:
            indexer[axis] = slice(i, i + 2)
            j = i + 1
            weights = np.array([(j - index), (index - i)], float)
            wshape = [1] * sorted.ndim
            wshape[axis] = 2
            weights.shape = wshape
            sumval = weights.sum()

        # Use add.reduce in both cases to coerce data type as well as
        # check and use out array.
        return np.add.reduce(sorted[indexer] * weights, axis=axis, out=out) / sumval


class Image(object):

    """This class defines images. As such, it contains data of the different
    *channels* of the image (red, green, and blue for example). The *mode*
    tells if the channels define a black and white image ("L"), an rgb image
    ("RGB"), an YCbCr image ("YCbCr"), or an indexed image ("P"), in which case
    a *palette* is needed. Each mode has also a corresponding alpha mode, which
    is the mode with an "A" in the end: for example "RGBA" is rgb with an alpha
    channel. *fill_value* sets how the image is filled where data is missing,
    since channels are numpy masked arrays. Setting it to (0,0,0) in RGB mode
    for example will produce black where data is missing."None" (default) will
    produce transparency (thus adding an alpha channel) if the file format
    allows it, black otherwise.

    The channels are considered to contain floating point values in the range
    [0.0,1.0]. In order to normalize the input data, the *color_range*
    parameter defines the original range of the data. The conversion to the
    classical [0,255] range and byte type is done automagically when saving the
    image to file.
    """
    modes = ["L", "LA", "RGB", "RGBA", "YCbCr", "YCbCrA", "P", "PA"]

    def __init__(self, channels=None, mode="L", color_range=None,
                 fill_value=None, palette=None):

        self.channels = None
        self.mode = None
        self.width = 0
        self.height = 0
        self.fill_value = None
        self.palette = None
        self.shape = None
        self.info = {}

        self._secondary_mode = "RGB"

        if(channels is not None and
           not isinstance(channels, (tuple, set, list,
                                     np.ndarray, np.ma.core.MaskedArray))):
            raise TypeError("Channels should a tuple, set, list, numpy array, "
                            "or masked array.")

        if(isinstance(channels, (tuple, list)) and
           len(channels) != len(re.findall("[A-Z]", mode))):
            raise ValueError("Number of channels does not match mode.")

        if mode not in self.modes:
            raise ValueError("Unknown mode.")

        if(color_range is not None and
           not _is_pair(color_range) and
           not _is_list_of_pairs(color_range)):
            raise ValueError("Color_range should be a pair"
                             " or a list/tuple/set of pairs.")
        if(color_range is not None and
           _is_list_of_pairs(color_range) and
           (channels is None or
                len(color_range) != len(channels))):
            raise ValueError("Color_range length does not match number of "
                             "channels.")

        if(color_range is not None and
           (((mode == "L" or mode == "P") and not _is_pair(color_range)) and
                (len(color_range) != len(re.findall("[A-Z]", mode))))):
            raise ValueError("Color_range does not match mode")

        self.mode = mode

        if isinstance(fill_value, (tuple, list, set)):
            self.fill_value = list(fill_value)
        elif fill_value is not None:
            self.fill_value = [fill_value]
        else:
            self.fill_value = None

        self.channels = []
        self.palette = palette

        if isinstance(channels, (tuple, list)):
            if _areinstances(channels, (np.ma.core.MaskedArray, np.ndarray,
                                        list, tuple)):
                for i, chn in enumerate(channels):
                    if color_range is not None:
                        color_min = color_range[i][0]
                        color_max = color_range[i][1]
                    else:
                        color_min = 0.0
                        color_max = 1.0

                    # Add data to image object as a channel
                    self._add_channel(chn, color_min, color_max)

                    self.shape = self.channels[-1].shape
                    if self.shape != self.channels[0].shape:
                        raise ValueError("Channels must have the same shape.")
                self.height = self.shape[0]
                self.width = self.shape[1]
            else:
                raise ValueError("Channels must all be arrays, lists or "
                                 "tuples.")
        elif channels is not None:
            self.height = channels.shape[0]
            self.width = channels.shape[1]
            self.shape = channels.shape

            if color_range is not None:
                color_min = color_range[0]
                color_max = color_range[1]
            else:
                color_min = 0.0
                color_max = 1.0

            # Add data to image object as a channel
            self._add_channel(channels, color_min, color_max)

        else:
            self.shape = (0, 0)
            self.width = 0
            self.height = 0

    def _add_channel(self, chn, color_min, color_max):
        """Adds a channel to the image object
        """

        if isinstance(chn, np.ma.core.MaskedArray):
            chn_data = chn.data
            chn_mask = chn.mask
        else:
            chn_data = np.array(chn)
            chn_mask = False
        scaled = ((chn_data - color_min) *
                  1.0 / (color_max - color_min))
        self.channels.append(np.ma.array(scaled, mask=chn_mask))

    def _finalize(self, dtype=np.uint8):
        """Finalize the image, that is put it in RGB mode, and set the channels
        in 8bit format ([0,255] range).
        """
        channels = []
        if self.mode == "P":
            self.convert("RGB")
        if self.mode == "PA":
            self.convert("RGBA")

        for chn in self.channels:
            if isinstance(chn, np.ma.core.MaskedArray):
                final_data = chn.data.clip(0, 1) * np.iinfo(dtype).max
            else:
                final_data = chn.clip(0, 1) * np.iinfo(dtype).max

            channels.append(np.ma.array(final_data,
                                        dtype,
                                        mask=chn.mask))
        if self.fill_value is not None:
            fill_value = [int(col * np.iinfo(dtype).max)
                          for col in self.fill_value]
        else:
            fill_value = None
        return channels, fill_value

    def is_empty(self):
        """Checks for an empty image.
        """
        if(((self.channels == []) and (not self.shape == (0, 0))) or
           ((not self.channels == []) and (self.shape == (0, 0)))):
            raise RuntimeError("Channels-shape mismatch.")
        return self.channels == [] and self.shape == (0, 0)

    def show(self):
        """Display the image on screen.
        """
        self.pil_image().show()

    def pil_image(self):
        """Return a PIL image from the current image.
        """
        channels, fill_value = self._finalize()

        if self.is_empty():
            return Pil.new(self.mode, (0, 0))

        if(self.mode == "L"):
            if fill_value is not None:
                img = Pil.fromarray(channels[0].filled(fill_value))
            else:
                img = Pil.fromarray(channels[0].filled(0))
                alpha = np.zeros(channels[0].shape, np.uint8)
                mask = np.ma.getmaskarray(channels[0])
                alpha = np.where(mask, alpha, 255)
                pil_alpha = Pil.fromarray(alpha)

                img = Pil.merge("LA", (img, pil_alpha))
        elif(self.mode == "LA"):
            if fill_value is not None:
                img = Pil.fromarray(channels[0].filled(fill_value))
                pil_alpha = Pil.fromarray(channels[1])
            else:
                img = Pil.fromarray(channels[0].filled(0))
                alpha = np.zeros(channels[0].shape, np.uint8)
                mask = np.ma.getmaskarray(channels[0])
                alpha = np.where(mask, alpha, channels[1])
                pil_alpha = Pil.fromarray(alpha)
            img = Pil.merge("LA", (img, pil_alpha))

        elif(self.mode == "RGB"):
            # Mask where all channels have missing data (incomplete data will
            # be shown).
            mask = (np.ma.getmaskarray(channels[0]) &
                    np.ma.getmaskarray(channels[1]) &
                    np.ma.getmaskarray(channels[2]))

            if fill_value is not None:
                pil_r = Pil.fromarray(channels[0].filled(fill_value[0]))
                pil_g = Pil.fromarray(channels[1].filled(fill_value[1]))
                pil_b = Pil.fromarray(channels[2].filled(fill_value[2]))
                img = Pil.merge("RGB", (pil_r, pil_g, pil_b))
            else:
                pil_r = Pil.fromarray(channels[0].filled(0))
                pil_g = Pil.fromarray(channels[1].filled(0))
                pil_b = Pil.fromarray(channels[2].filled(0))

                alpha = np.zeros(channels[0].shape, np.uint8)
                alpha = np.where(mask, alpha, 255)
                pil_a = Pil.fromarray(alpha)

                img = Pil.merge("RGBA", (pil_r, pil_g, pil_b, pil_a))

        elif(self.mode == "RGBA"):
            # Mask where all channels have missing data (incomplete data will
            # be shown).
            mask = (np.ma.getmaskarray(channels[0]) &
                    np.ma.getmaskarray(channels[1]) &
                    np.ma.getmaskarray(channels[2]) &
                    np.ma.getmaskarray(channels[3]))

            if fill_value is not None:
                pil_r = Pil.fromarray(channels[0].filled(fill_value[0]))
                pil_g = Pil.fromarray(channels[1].filled(fill_value[1]))
                pil_b = Pil.fromarray(channels[2].filled(fill_value[2]))
                pil_a = Pil.fromarray(channels[3].filled(fill_value[3]))
                img = Pil.merge("RGBA", (pil_r, pil_g, pil_b, pil_a))
            else:
                pil_r = Pil.fromarray(channels[0].filled(0))
                pil_g = Pil.fromarray(channels[1].filled(0))
                pil_b = Pil.fromarray(channels[2].filled(0))

                alpha = np.where(mask, 0, channels[3])
                pil_a = Pil.fromarray(alpha)

                img = Pil.merge("RGBA", (pil_r, pil_g, pil_b, pil_a))

        else:
            raise TypeError("Does not know how to use mode %s." % (self.mode))

        return img

    def save(self, filename, compression=6, fformat=None):
        """Save the image to the given *filename*. For some formats like jpg
        and png, the work is delegated to :meth:`pil_save`, which doesn't
        support the *compression* option.
        """
        self.pil_save(filename, compression, fformat)

    def pil_save(self, filename, compression=6, fformat=None):
        """Save the image to the given *filename* using PIL. For now, the
        compression level [0-9] is ignored, due to PIL's lack of support. See
        also :meth:`save`.
        """
        # PIL does not support compression option.
        del compression

        if self.is_empty():
            raise IOError("Cannot save an empty image")

        ensure_dir(filename)

        fformat = fformat or os.path.splitext(filename)[1][1:4]
        fformat = check_image_format(fformat)

        params = {}

        if fformat == 'png':
            # Take care of GeoImage.tags.
            params['pnginfo'] = self._pngmeta()

        self.pil_image().save(filename, fformat, **params)

    def _pngmeta(self):
        """It will return GeoImage.tags as a PNG metadata object.

         Inspired by:
         public domain, Nick Galbreath
         http://blog.modp.com/2007/08/python-pil-and-png-metadata-take-2.html
        """
        reserved = ('interlace', 'gamma', 'dpi', 'transparency', 'aspect')

        try:
            tags = self.tags
        except AttributeError:
            tags = {}

        # Undocumented class
        from PIL import PngImagePlugin
        meta = PngImagePlugin.PngInfo()

        # Copy from tags to new dict
        for k__, v__ in tags.items():
            if k__ not in reserved:
                meta.add_text(k__, v__, 0)

        return meta

    def putalpha(self, alpha):
        """Adds an *alpha* channel to the current image, or replaces it with
        *alpha* if it already exists.
        """
        alpha = np.ma.array(alpha)
        if(not (alpha.shape[0] == 0 and
                self.shape[0] == 0) and
           alpha.shape != self.shape):
            raise ValueError("Alpha channel shape should match image shape")

        if(not self.mode.endswith("A")):
            self.convert(self.mode + "A")
        if not self.is_empty():
            self.channels[-1] = alpha

    def _rgb2ycbcr(self, mode):
        """Convert the image from RGB mode to YCbCr."""

        self._check_modes(("RGB", "RGBA"))

        (self.channels[0], self.channels[1], self.channels[2]) = \
            rgb2ycbcr(self.channels[0],
                      self.channels[1],
                      self.channels[2])

        if self.fill_value is not None:
            self.fill_value[0:3] = rgb2ycbcr(self.fill_value[0],
                                             self.fill_value[1],
                                             self.fill_value[2])

        self.mode = mode

    def _ycbcr2rgb(self, mode):
        """Convert the image from YCbCr mode to RGB.
        """

        self._check_modes(("YCbCr", "YCbCrA"))

        (self.channels[0], self.channels[1], self.channels[2]) = \
            ycbcr2rgb(self.channels[0],
                      self.channels[1],
                      self.channels[2])

        if self.fill_value is not None:
            self.fill_value[0:3] = ycbcr2rgb(self.fill_value[0],
                                             self.fill_value[1],
                                             self.fill_value[2])

        self.mode = mode

    def _to_p(self, mode):
        """Convert the image to P or PA mode.
        """

        if self.mode.endswith("A"):
            chans = self.channels[:-1]
            alpha = self.channels[-1]
            self._secondary_mode = self.mode[:-1]
        else:
            chans = self.channels
            alpha = None
            self._secondary_mode = self.mode

        palette = []
        selfmask = reduce(np.ma.mask_or, [chn.mask for chn in chans])
        new_chn = np.ma.zeros(self.shape, dtype=int)
        color_nb = 0

        for i in range(self.height):
            for j in range(self.width):
                current_col = tuple([chn[i, j] for chn in chans])
                try:
                    (idx
                     for idx in range(len(palette))
                     if palette[idx] == current_col).next()
                except StopIteration:
                    idx = color_nb
                    palette.append(current_col)
                    color_nb = color_nb + 1

                new_chn[i, j] = idx

        if self.fill_value is not None:
            if self.mode.endswith("A"):
                current_col = tuple(self.fill_value[:-1])
                fill_alpha = [self.fill_value[-1]]
            else:
                current_col = tuple(self.fill_value)
                fill_alpha = []
            try:
                (idx
                 for idx in range(len(palette))
                 if palette[idx] == current_col).next()
            except StopIteration:
                idx = color_nb
                palette.append(current_col)
                color_nb = color_nb + 1

            self.fill_value = [idx] + fill_alpha

        new_chn.mask = selfmask
        self.palette = palette
        if alpha is None:
            self.channels = [new_chn]
        else:
            self.channels = [new_chn, alpha]
        self.mode = mode

    def _from_p(self, mode):
        """Convert the image from P or PA mode.
        """

        self._check_modes(("P", "PA"))

        if self.mode.endswith("A"):
            alpha = self.channels[-1]
        else:
            alpha = None

        chans = []
        cdfs = []

        color_chan = self.channels[0]

        for i in range(len(self.palette[0])):
            cdfs.append(np.zeros(len(self.palette)))
            for j in range(len(self.palette)):
                cdfs[i][j] = self.palette[j][i]
            new_chn = np.ma.array(np.interp(color_chan,
                                            np.arange(len(self.palette)),
                                            cdfs[i]),
                                  mask=color_chan.mask)
            chans.append(new_chn)

        if self.fill_value is not None:
            if alpha is not None:
                fill_alpha = self.fill_value[-1]
                self.fill_value = list(self.palette[int(self.fill_value[0])])
                self.fill_value += [fill_alpha]
            else:
                self.fill_value = list(self.palette[int(self.fill_value[0])])

        self.mode = self._secondary_mode
        self.channels = chans
        if alpha is not None:
            self.channels.append(alpha)
            self.mode = self.mode + "A"

        self.convert(mode)

    def _check_modes(self, modes):
        """Check that the image is in on of the given *modes*, raise an
        exception otherwise.
        """
        if not isinstance(modes, (tuple, list, set)):
            modes = [modes]
        if self.mode not in modes:
            raise ValueError("Image not in suitable mode: %s" % modes)

    def _l2rgb(self, mode):
        """Convert from L (black and white) to RGB.
        """
        self._check_modes(("L", "LA"))
        self.channels.append(self.channels[0].copy())
        self.channels.append(self.channels[0].copy())
        if self.fill_value is not None:
            self.fill_value = self.fill_value[:1] * 3 + self.fill_value[1:]
        if self.mode == "LA":
            self.channels[1], self.channels[3] = \
                self.channels[3], self.channels[1]
        self.mode = mode

    def _rgb2l(self, mode):
        """Convert from RGB to monochrome L.
        """
        self._check_modes(("RGB", "RGBA"))

        kb_ = 0.114
        kr_ = 0.299

        r__ = self.channels[0]
        g__ = self.channels[1]
        b__ = self.channels[2]

        y__ = kr_ * r__ + (1 - kr_ - kb_) * g__ + kb_ * b__

        if self.fill_value is not None:
            self.fill_value = ([rgb2ycbcr(self.fill_value[0],
                                          self.fill_value[1],
                                          self.fill_value[2])[0]] +
                               self.fill_value[3:])

        self.channels = [y__] + self.channels[3:]

        self.mode = mode

    def _ycbcr2l(self, mode):
        """Convert from YCbCr to L.
        """
        self._check_modes(("YCbCr", "YCbCrA"))

        self.channels = [self.channels[0]] + self.channels[3:]
        if self.fill_value is not None:
            self.fill_value = [self.fill_value[0]] + self.fill_value[3:]
        self.mode = mode

    def _l2ycbcr(self, mode):
        """Convert from L to YCbCr.
        """
        self._check_modes(("L", "LA"))

        luma = self.channels[0]
        zeros = np.ma.zeros(luma.shape)
        zeros.mask = luma.mask

        self.channels = [luma, zeros, zeros] + self.channels[1:]

        if self.fill_value is not None:
            self.fill_value = [self.fill_value[0], 0, 0] + self.fill_value[1:]

        self.mode = mode

    def convert(self, mode):
        """Convert the current image to the given *mode*. See :class:`Image`
        for a list of available modes.
        """
        if mode == self.mode:
            return

        if mode not in ["L", "LA", "RGB", "RGBA",
                        "YCbCr", "YCbCrA", "P", "PA"]:
            raise ValueError("Mode %s not recognized." % (mode))

        if self.is_empty():
            self.mode = mode
            return

        if(mode == self.mode + "A"):
            self.channels.append(np.ma.ones(self.channels[0].shape))
            if self.fill_value is not None:
                self.fill_value += [1]
            self.mode = mode

        elif(mode + "A" == self.mode):
            self.channels = self.channels[:-1]
            if self.fill_value is not None:
                self.fill_value = self.fill_value[:-1]
            self.mode = mode

        elif(mode.endswith("A") and not self.mode.endswith("A")):
            self.convert(self.mode + "A")
            self.convert(mode)

        elif(self.mode.endswith("A") and not mode.endswith("A")):
            self.convert(self.mode[:-1])
            self.convert(mode)

        else:
            cases = {
                "RGB": {"YCbCr": self._rgb2ycbcr,
                        "L": self._rgb2l,
                        "P": self._to_p},
                "RGBA": {"YCbCrA": self._rgb2ycbcr,
                         "LA": self._rgb2l,
                         "PA": self._to_p},
                "YCbCr": {"RGB": self._ycbcr2rgb,
                          "L": self._ycbcr2l,
                          "P": self._to_p},
                "YCbCrA": {"RGBA": self._ycbcr2rgb,
                           "LA": self._ycbcr2l,
                           "PA": self._to_p},
                "L": {"RGB": self._l2rgb,
                      "YCbCr": self._l2ycbcr,
                      "P": self._to_p},
                "LA": {"RGBA": self._l2rgb,
                       "YCbCrA": self._l2ycbcr,
                       "PA": self._to_p},
                "P": {"RGB": self._from_p,
                      "YCbCr": self._from_p,
                      "L": self._from_p},
                "PA": {"RGBA": self._from_p,
                       "YCbCrA": self._from_p,
                       "LA": self._from_p}}
            try:
                cases[self.mode][mode](mode)
            except KeyError:
                raise ValueError("Conversion from %s to %s not implemented !"
                                 % (self.mode, mode))

    def clip(self, channels=True):
        """Limit the values of the array to the default [0,1] range. *channels*
        says which channels should be clipped."""
        if not (isinstance(channels, (tuple, list))):
            channels = [channels] * len(self.channels)

        for i in range(len(self.channels)):
            if channels[i]:
                self.channels[i] = np.ma.clip(self.channels[i], 0.0, 1.0)

    def resize(self, shape):
        """Resize the image to the given *shape* tuple, in place. For zooming,
        nearest neighbour method is used, while for shrinking, decimation is
        used. Therefore, *shape* must be a multiple or a divisor of the image
        shape.
        """
        if self.is_empty():
            raise ValueError("Cannot resize an empty image")

        factor = [1, 1]
        zoom = [True, True]
        zoom[0] = shape[0] >= self.height
        zoom[1] = shape[1] >= self.width

        if zoom[0]:
            factor[0] = shape[0] * 1.0 / self.height
        else:
            factor[0] = self.height * 1.0 / shape[0]
        if zoom[1]:
            factor[1] = shape[1] * 1.0 / self.width
        else:
            factor[1] = self.width * 1.0 / shape[1]

        if(int(factor[0]) != factor[0] or
           int(factor[1]) != factor[1]):
            raise ValueError("Resize not of integer factor!")

        factor[0] = int(factor[0])
        factor[1] = int(factor[1])

        i = 0
        for chn in self.channels:
            if zoom[0]:
                chn = chn.repeat([factor[0]] * chn.shape[0], axis=0)
            else:
                chn = chn[[idx * factor[0]
                           for idx in range(self.height / factor[0])],
                          :]
            if zoom[1]:
                self.channels[i] = chn.repeat([factor[1]] * chn.shape[1],
                                              axis=1)
            else:
                self.channels[i] = chn[:,
                                       [idx * factor[1]
                                        for idx in range(self.width /
                                                         factor[1])]]

            i = i + 1

        self.height = self.channels[0].shape[0]
        self.width = self.channels[0].shape[1]
        self.shape = self.channels[0].shape

    def replace_luminance(self, luminance):
        """Replace the Y channel of the image by the array *luminance*. If the
        image is not in YCbCr mode, it is converted automatically to and
        from that mode.
        """
        if self.is_empty():
            return

        if (luminance.shape != self.channels[0].shape):
            if ((luminance.shape[0] * 1.0 / luminance.shape[1]) ==
                    (self.channels[0].shape[0] * 1.0 / self.channels[0].shape[1])):
                if luminance.shape[0] > self.channels[0].shape[0]:
                    self.resize(luminance.shape)
                else:
                    raise NameError("Luminance smaller than the image !")
            else:
                raise NameError("Not the good shape !")

        mode = self.mode
        if mode.endswith("A"):
            self.convert("YCbCrA")
            self.channels[0] = luminance
            self.convert(mode)
        else:
            self.convert("YCbCr")
            self.channels[0] = luminance
            self.convert(mode)

    def enhance(self, inverse=False, gamma=1.0, stretch="no"):
        """Image enhancement function. It applies **in this order** inversion,
        gamma correction, and stretching to the current image, with parameters
        *inverse* (see :meth:`Image.invert`), *gamma* (see
        :meth:`Image.gamma`), and *stretch* (see :meth:`Image.stretch`).
        """
        self.invert(inverse)
        self.gamma(gamma)
        self.stretch(stretch)

    def gamma(self, gamma=1.0):
        """Apply gamma correction to the channels of the image. If *gamma* is a
        tuple, then it should have as many elements as the channels of the
        image, and the gamma correction is applied elementwise. If *gamma* is a
        number, the same gamma correction is applied on every channel, if there
        are several channels in the image. The behaviour of :func:`gamma` is
        undefined outside the normal [0,1] range of the channels.
        """
        if not isinstance(gamma, (int, long, float)):
            if(not isinstance(gamma, (tuple, list, set)) or
               not _areinstances(gamma, (int, long, float))):
                raise TypeError("Gamma should be a real number, or an iterable "
                                "of real numbers.")

        if(isinstance(gamma, (list, tuple, set)) and
           len(gamma) != len(self.channels)):
            raise ValueError("Number of channels and gamma components differ.")

        if gamma < 0:
            raise ValueError("Gamma correction must be a positive number.")

        if gamma == 1.0:
            return
        if (isinstance(gamma, (tuple, list))):
            gamma_list = list(gamma)
        else:
            gamma_list = [gamma] * len(self.channels)
        for i in range(len(self.channels)):
            if(isinstance(self.channels[i], np.ma.core.MaskedArray)):
                if ne:
                    self.channels[i] = np.ma.array(
                        ne.evaluate("data ** (1.0 / gamma)",
                                    local_dict={"data": self.channels[i].data,
                                                'gamma': gamma_list[i]}),
                        mask=self.channels[i].mask,
                        copy=False)
                else:
                    self.channels[i] = np.ma.array(self.channels[i].data **
                                                   (1.0 / gamma_list[i]),
                                                   mask=self.channels[i].mask,
                                                   copy=False)
            else:
                self.channels[i] = np.where(self.channels[i] >= 0,
                                            self.channels[i] **
                                            (1.0 / gamma_list[i]),
                                            self.channels[i])

    def stretch(self, stretch="no", **kwarg):
        """Apply stretching to the current image. The value of *stretch* sets
        the type of stretching applied. The values "histogram", "linear",
        "crude" (or "crude-stretch") perform respectively histogram
        equalization, contrast stretching (with 5% cutoff on both sides), and
        contrast stretching without cutoff. The value "logarithmic" or "log"
        will do a logarithmic enhancement towards white. If a tuple or a list
        of two values is given as input, then a contrast stretching is performed
        with the values as cutoff. These values should be normalized in the
        range [0.0,1.0].
        """
        if((isinstance(stretch, tuple) or
                isinstance(stretch, list))):
            if len(stretch) == 2:
                for i in range(len(self.channels)):
                    self.stretch_linear(i, cutoffs=stretch, **kwarg)
            else:
                raise ValueError(
                    "Stretch tuple must have exactly two elements")
        elif stretch == "linear":
            for i in range(len(self.channels)):
                self.stretch_linear(i, **kwarg)
        elif stretch == "histogram":
            for i in range(len(self.channels)):
                self.stretch_hist_equalize(i, **kwarg)
        elif(stretch in ["crude", "crude-stretch"]):
            for i in range(len(self.channels)):
                self.crude_stretch(i, **kwarg)
        elif(stretch in ["log", "logarithmic"]):
            for i in range(len(self.channels)):
                self.stretch_logarithmic(i, **kwarg)
        elif(stretch == "no"):
            return
        elif isinstance(stretch, str):
            raise ValueError("Stretching method %s not recognized." % stretch)
        else:
            raise TypeError("Stretch parameter must be a string or a tuple.")

    def invert(self, invert=True):
        """Inverts all the channels of a image according to *invert*. If invert
        is a tuple or a list, elementwise invertion is performed, otherwise all
        channels are inverted if *invert* is true (default).
        """
        if(isinstance(invert, (tuple, list, set)) and
           len(self.channels) != len(invert)):
            raise ValueError(
                "Number of channels and invert components differ.")

        if isinstance(invert, (tuple, list, set)):
            i = 0
            for chn in self.channels:
                if(invert[i]):
                    self.channels[i] = 1.0 - chn
                i = i + 1
        elif(invert):
            i = 0
            for chn in self.channels:
                self.channels[i] = 1.0 - chn
                i = i + 1

    def stretch_hist_equalize(self, ch_nb):
        """Stretch the current image's colors by performing histogram
        equalization on channel *ch_nb*.
        """
        LOG.info("Perform a histogram equalized contrast stretch.")

        if(self.channels[ch_nb].size ==
           np.ma.count_masked(self.channels[ch_nb])):
            LOG.warning("Nothing to stretch !")
            return

        arr = self.channels[ch_nb]

        nwidth = 2048.0

        carr = arr.compressed()

        cdf = np.arange(0.0, 1.0, 1 / nwidth)
        LOG.debug("Make histogram bins having equal amount of data, " +
                  "using numpy percentile function:")
        bins = percentile(carr, list(cdf * 100))

        res = np.ma.empty_like(arr)
        res.mask = np.ma.getmaskarray(arr)
        res[~res.mask] = np.interp(carr, bins, cdf)

        self.channels[ch_nb] = res

    def stretch_logarithmic(self, ch_nb, factor=100.):
        """Move data into range [1:factor] and do a normalized logarithmic
        enhancement.
        """
        LOG.debug("Perform a logarithmic contrast stretch.")
        if ((self.channels[ch_nb].size ==
             np.ma.count_masked(self.channels[ch_nb])) or
                (self.channels[ch_nb].min() == self.channels[ch_nb].max())):
            LOG.warning("Nothing to stretch !")
            return

        crange = (0., 1.0)

        arr = self.channels[ch_nb]
        b = float(crange[1] - crange[0]) / np.log(factor)
        c = float(crange[0])
        slope = (factor - 1.) / float(arr.max() - arr.min())
        arr = 1. + (arr - arr.min()) * slope
        arr = c + b * np.log(arr)
        self.channels[ch_nb] = arr

    def stretch_linear(self, ch_nb, cutoffs=(0.005, 0.005)):
        """Stretch linearly the contrast of the current image on channel
        *ch_nb*, using *cutoffs* for left and right trimming.
        """
        LOG.debug("Perform a linear contrast stretch.")

        if((self.channels[ch_nb].size ==
            np.ma.count_masked(self.channels[ch_nb])) or
           self.channels[ch_nb].min() == self.channels[ch_nb].max()):
            LOG.warning("Nothing to stretch !")
            return

        arr = self.channels[ch_nb]
        carr = arr.compressed()

        LOG.debug("Calculate the histogram percentiles: ")
        LOG.debug("Left and right percentiles: " +
                  str(cutoffs[0] * 100) + " " + str(cutoffs[1] * 100))

        left, right = percentile(carr, [cutoffs[0] * 100,
                                        100. - cutoffs[1] * 100])

        delta_x = (right - left)
        LOG.debug("Interval: left=%f, right=%f width=%f"
                  % (left, right, delta_x))

        if delta_x > 0.0:
            self.channels[ch_nb] = np.ma.array((arr - left) / delta_x,
                                               mask=arr.mask)
        else:
            self.channels[ch_nb] = np.ma.zeros(arr.shape)
            LOG.warning("Unable to make a contrast stretch!")

    def crude_stretch(self, ch_nb, min_stretch=None, max_stretch=None):
        """Perform simple linear stretching (without any cutoff) on the channel
        *ch_nb* of the current image and normalize to the [0,1] range."""

        if(min_stretch is None):
            min_stretch = self.channels[ch_nb].min()
        if(max_stretch is None):
            max_stretch = self.channels[ch_nb].max()

        if((not self.channels[ch_nb].mask.all()) and
                max_stretch - min_stretch > 0):
            stretched = self.channels[ch_nb].data.astype(np.float)
            stretched -= min_stretch
            stretched /= max_stretch - min_stretch
            self.channels[ch_nb] = np.ma.array(stretched,
                                               mask=self.channels[ch_nb].mask,
                                               copy=False)
        else:
            LOG.warning("Nothing to stretch !")

    def merge(self, img):
        """Use the provided image as background for the current *img* image,
        that is if the current image has missing data.
        """
        if self.is_empty():
            raise ValueError("Cannot merge an empty image.")

        if(self.mode != img.mode):
            raise ValueError("Cannot merge image of different modes.")

        selfmask = reduce(np.ma.mask_or, [chn.mask for chn in self.channels])

        for i in range(len(self.channels)):
            self.channels[i] = np.ma.where(selfmask,
                                           img.channels[i],
                                           self.channels[i])
            self.channels[i].mask = np.logical_and(selfmask,
                                                   img.channels[i].mask)


def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True


def _areinstances(the_list, types):
    """Check if all the elements of the list are of given type.
    """
    return all([isinstance(item, types) for item in the_list])


def _is_pair(item):
    """Check if an item is a pair (tuple of size 2).
    """
    return (isinstance(item, (list, tuple, set)) and
            len(item) == 2 and
            not isinstance(item[0], (list, tuple, set)) and
            not isinstance(item[1], (list, tuple, set)))


def _is_list_of_pairs(the_list):
    """Check if a list contains only pairs.
    """
    return all([_is_pair(item) for item in the_list])


def ycbcr2rgb(y__, cb_, cr_):
    """Convert the three YCbCr channels to RGB channels.
    """

    kb_ = 0.114
    kr_ = 0.299

    r__ = 2 * cr_ / (1 - kr_) + y__
    b__ = 2 * cb_ / (1 - kb_) + y__
    g__ = (y__ - kr_ * r__ - kb_ * b__) / (1 - kr_ - kb_)

    return r__, g__, b__


def rgb2ycbcr(r__, g__, b__):
    """Convert the three RGB channels to YCbCr."""

    kb_ = 0.114
    kr_ = 0.299

    y__ = kr_ * r__ + (1 - kr_ - kb_) * g__ + kb_ * b__
    cb_ = 1. / (2 * (1 - kb_)) * (b__ - y__)
    cr_ = 1. / (2 * (1 - kr_)) * (r__ - y__)

    return y__, cb_, cr_
