#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2009.

# SMHI,
# Folkborgsvägen 1,
# Norrköping, 
# Sweden

# Author(s):
 
#   Martin Raspaud <martin.raspaud@smhi.se>
#   Adam Dybbroe <adam.dybbroe@smhi.se>

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

"""Palette holder module.
"""

import logging
LOG = logging.getLogger("palettes")


def tv_legend():
    """Palette for TV.
    """
    legend = []
    legend.append((  0,   0,   0)) # Unprocessed: Black
    legend.append((  0, 120,   0)) # Land
    legend.append((  0,   0, 215)) # Sea: Blue
    legend.append((  0, 120,   0)) # Land (Snow on land)
    legend.append((  0,   0, 215)) # Sea: Blue (Snow/Ice on sea)
        
    for i in range(5, 256):
        # All other pixel values are grey according to IR temp.        
        legend.append((i, i, i)) 
    
    return convert_palette(legend)

def vv_legend():
    """Palette for Swedish road authorities (Vägverket).
    """
    legend = []
    legend.append((  0,   0,   0)) # Unprocessed: Black
    legend.append((  0, 120,   0)) # Land
    legend.append((  0,   0, 215)) # Sea: Blue
    # Cloud type values 5 to 8:
    legend.append((255, 150,   0)) # Very low cumuliform
    legend.append((255, 100,   0)) # Very low
    legend.append((255, 220,   0)) # Low cumuliform
    legend.append((255, 180,   0)) # Low

    for i in range(7, 256):
        # All other pixel values are grey according to IR temp.        
        legend.append((i, i, i)) 
    
    return convert_palette(legend)

def cms_modified():
    """Palette for regular cloud classification.
    """
    legend = []
    legend.append((100, 100, 100)) # Unprocessed: Grey
    legend.append((  0, 120,   0))
    legend.append((  0,   0,   0)) # Sea: Black
    legend.append((250, 190, 250)) # Snow
    legend.append((220, 160, 220)) # Sea-ice
    
    legend.append((255, 150,   0)) # Very low cumuliform
    legend.append((255, 100,   0)) # Very low
    legend.append((255, 220,   0)) # Low cumuliform
    legend.append((255, 180,   0)) # Low
    legend.append((255, 255, 140)) # Medium cumuliform
    legend.append((240, 240,   0)) # Medium
    legend.append((250, 240, 200)) # High cumiliform
    legend.append((215, 215, 150)) # High
    legend.append((255, 255, 255)) # Very high cumuliform
    legend.append((230, 230, 230)) # Very high

    legend.append((  0,  80, 215)) # Semi-transparent thin
    legend.append((  0, 180, 230)) # Semi-transparent medium
    legend.append((  0, 240, 240)) # Semi-transparent thick
    legend.append(( 90, 200, 160)) # Semi-transparent above
    legend.append((200,   0, 200)) # Broken
    legend.append(( 95,  60,  30)) # Undefined: Brown
    
    return convert_palette(legend)

def ctth_height():
    """CTTH height palette.
    """
    legend = []    
    legend.append((0,     0,   0))
    legend.append((255,   0, 216)) # 0 meters
    legend.append((126,   0,  43))
    legend.append((153,  20,  47))
    legend.append((178,  51,   0))
    legend.append((255,  76,   0))
    legend.append((255, 102,   0))
    legend.append((255, 164,   0))
    legend.append((255, 216,   0))
    legend.append((216, 255,   0))
    legend.append((178, 255,   0))
    legend.append((153, 255,   0))
    legend.append((0,   255,   0))
    legend.append((0,   140,  48))
    legend.append((0,   178, 255))
    legend.append((0,   216, 255))
    legend.append((0,   255, 255))
    legend.append((238, 214, 210))
    legend.append((239, 239, 223))
    legend.append((255, 255, 255)) # 10,000 meters
    for i in range(79):
        legend.append((255, 255, 255)) 
    legend.append((224,224,224))
    
    return convert_palette(legend)

def chlorophyll_a():
    """Chlorophyll A colour palette for Modis and Meris
    products.
    """
    try:
        from matplotlib.pylab import cm
        legend = cm.spectral(range(256))
    except ImportError:
        LOG.info("No pylab installed. "+ 
                 "Will make the palette from hand")
        legend = []
        legend.append((0.000000, 0.000000, 0.000000))
        legend.append((9.370604, 0.000000, 10.707827))
        legend.append((18.741208, 0.000000, 21.415655))
        legend.append((28.111812, 0.000000, 32.123482))
        legend.append((37.482416, 0.000000, 42.831310))
        legend.append((46.853020, 0.000000, 53.539137))
        legend.append((56.223624, 0.000000, 64.246965))
        legend.append((65.594227, 0.000000, 74.954792))
        legend.append((74.964831, 0.000000, 85.662620))
        legend.append((84.335435, 0.000000, 96.370447))
        legend.append((93.706039, 0.000000, 107.078275))
        legend.append((103.076643, 0.000000, 117.786102))
        legend.append((112.447247, 0.000000, 128.493929))
        legend.append((119.809506, 0.000000, 136.859608))
        legend.append((121.146729, 0.000000, 138.198839))
        legend.append((122.483953, 0.000000, 139.538071))
        legend.append((123.821176, 0.000000, 140.877302))
        legend.append((125.158400, 0.000000, 142.216533))
        legend.append((126.495624, 0.000000, 143.555765))
        legend.append((127.832847, 0.000000, 144.894996))
        legend.append((129.170071, 0.000000, 146.234227))
        legend.append((130.507294, 0.000000, 147.573459))
        legend.append((131.844518, 0.000000, 148.912690))
        legend.append((133.181741, 0.000000, 150.251922))
        legend.append((134.518965, 0.000000, 151.591153))
        legend.append((135.856188, 0.000000, 152.930384))
        legend.append((131.170886, 0.000000, 154.269616))
        legend.append((120.463059, 0.000000, 155.608847))
        legend.append((109.755231, 0.000000, 156.948078))
        legend.append((99.047404, 0.000000, 158.287310))
        legend.append((88.339576, 0.000000, 159.626541))
        legend.append((77.631749, 0.000000, 160.965773))
        legend.append((66.923922, 0.000000, 162.305004))
        legend.append((56.216094, 0.000000, 163.644235))
        legend.append((45.508267, 0.000000, 164.983467))
        legend.append((34.800439, 0.000000, 166.322698))
        legend.append((24.092612, 0.000000, 167.661929))
        legend.append((13.384784, 0.000000, 169.001161))
        legend.append((2.676957, 0.000000, 170.340392))
        legend.append((0.000000, 0.000000, 173.686965))
        legend.append((0.000000, 0.000000, 177.702651))
        legend.append((0.000000, 0.000000, 181.718337))
        legend.append((0.000000, 0.000000, 185.734024))
        legend.append((0.000000, 0.000000, 189.749710))
        legend.append((0.000000, 0.000000, 193.765396))
        legend.append((0.000000, 0.000000, 197.781082))
        legend.append((0.000000, 0.000000, 201.796769))
        legend.append((0.000000, 0.000000, 205.812455))
        legend.append((0.000000, 0.000000, 209.828141))
        legend.append((0.000000, 0.000000, 213.843827))
        legend.append((0.000000, 0.000000, 217.859514))
        legend.append((0.000000, 0.000000, 221.875200))
        legend.append((0.000000, 9.370604, 221.875200))
        legend.append((0.000000, 18.741208, 221.875200))
        legend.append((0.000000, 28.111812, 221.875200))
        legend.append((0.000000, 37.482416, 221.875200))
        legend.append((0.000000, 46.853020, 221.875200))
        legend.append((0.000000, 56.223624, 221.875200))
        legend.append((0.000000, 65.594227, 221.875200))
        legend.append((0.000000, 74.964831, 221.875200))
        legend.append((0.000000, 84.335435, 221.875200))
        legend.append((0.000000, 93.706039, 221.875200))
        legend.append((0.000000, 103.076643, 221.875200))
        legend.append((0.000000, 112.447247, 221.875200))
        legend.append((0.000000, 120.144314, 221.875200))
        legend.append((0.000000, 122.820769, 221.875200))
        legend.append((0.000000, 125.497224, 221.875200))
        legend.append((0.000000, 128.173678, 221.875200))
        legend.append((0.000000, 130.850133, 221.875200))
        legend.append((0.000000, 133.526588, 221.875200))
        legend.append((0.000000, 136.203043, 221.875200))
        legend.append((0.000000, 138.879498, 221.875200))
        legend.append((0.000000, 141.555953, 221.875200))
        legend.append((0.000000, 144.232408, 221.875200))
        legend.append((0.000000, 146.908863, 221.875200))
        legend.append((0.000000, 149.585318, 221.875200))
        legend.append((0.000000, 152.261773, 221.875200))
        legend.append((0.000000, 154.269616, 219.867357))
        legend.append((0.000000, 155.608847, 215.851671))
        legend.append((0.000000, 156.948078, 211.835984))
        legend.append((0.000000, 158.287310, 207.820298))
        legend.append((0.000000, 159.626541, 203.804612))
        legend.append((0.000000, 160.965773, 199.788925))
        legend.append((0.000000, 162.305004, 195.773239))
        legend.append((0.000000, 163.644235, 191.757553))
        legend.append((0.000000, 164.983467, 187.741867))
        legend.append((0.000000, 166.322698, 183.726180))
        legend.append((0.000000, 167.661929, 179.710494))
        legend.append((0.000000, 169.001161, 175.694808))
        legend.append((0.000000, 170.340392, 171.679122))
        legend.append((0.000000, 170.675200, 168.666353))
        legend.append((0.000000, 170.675200, 165.987890))
        legend.append((0.000000, 170.675200, 163.309427))
        legend.append((0.000000, 170.675200, 160.630965))
        legend.append((0.000000, 170.675200, 157.952502))
        legend.append((0.000000, 170.675200, 155.274039))
        legend.append((0.000000, 170.675200, 152.595576))
        legend.append((0.000000, 170.675200, 149.917114))
        legend.append((0.000000, 170.675200, 147.238651))
        legend.append((0.000000, 170.675200, 144.560188))
        legend.append((0.000000, 170.675200, 141.881725))
        legend.append((0.000000, 170.675200, 139.203263))
        legend.append((0.000000, 170.675200, 136.524800))
        legend.append((0.000000, 169.335969, 125.816973))
        legend.append((0.000000, 167.996737, 115.109145))
        legend.append((0.000000, 166.657506, 104.401318))
        legend.append((0.000000, 165.318275, 93.693490))
        legend.append((0.000000, 163.979043, 82.985663))
        legend.append((0.000000, 162.639812, 72.277835))
        legend.append((0.000000, 161.300580, 61.570008))
        legend.append((0.000000, 159.961349, 50.862180))
        legend.append((0.000000, 158.622118, 40.154353))
        legend.append((0.000000, 157.282886, 29.446525))
        legend.append((0.000000, 155.943655, 18.738698))
        legend.append((0.000000, 154.604424, 8.030871))
        legend.append((0.000000, 154.269114, 0.000000))
        legend.append((0.000000, 156.945569, 0.000000))
        legend.append((0.000000, 159.622024, 0.000000))
        legend.append((0.000000, 162.298478, 0.000000))
        legend.append((0.000000, 164.974933, 0.000000))
        legend.append((0.000000, 167.651388, 0.000000))
        legend.append((0.000000, 170.327843, 0.000000))
        legend.append((0.000000, 173.004298, 0.000000))
        legend.append((0.000000, 175.680753, 0.000000))
        legend.append((0.000000, 178.357208, 0.000000))
        legend.append((0.000000, 181.033663, 0.000000))
        legend.append((0.000000, 183.710118, 0.000000))
        legend.append((0.000000, 186.386573, 0.000000))
        legend.append((0.000000, 189.064031, 0.000000))
        legend.append((0.000000, 191.742494, 0.000000))
        legend.append((0.000000, 194.420957, 0.000000))
        legend.append((0.000000, 197.099420, 0.000000))
        legend.append((0.000000, 199.777882, 0.000000))
        legend.append((0.000000, 202.456345, 0.000000))
        legend.append((0.000000, 205.134808, 0.000000))
        legend.append((0.000000, 207.813271, 0.000000))
        legend.append((0.000000, 210.491733, 0.000000))
        legend.append((0.000000, 213.170196, 0.000000))
        legend.append((0.000000, 215.848659, 0.000000))
        legend.append((0.000000, 218.527122, 0.000000))
        legend.append((0.000000, 221.205584, 0.000000))
        legend.append((0.000000, 223.882541, 0.000000))
        legend.append((0.000000, 226.558996, 0.000000))
        legend.append((0.000000, 229.235451, 0.000000))
        legend.append((0.000000, 231.911906, 0.000000))
        legend.append((0.000000, 234.588361, 0.000000))
        legend.append((0.000000, 237.264816, 0.000000))
        legend.append((0.000000, 239.941271, 0.000000))
        legend.append((0.000000, 242.617725, 0.000000))
        legend.append((0.000000, 245.294180, 0.000000))
        legend.append((0.000000, 247.970635, 0.000000))
        legend.append((0.000000, 250.647090, 0.000000))
        legend.append((0.000000, 253.323545, 0.000000))
        legend.append((0.000000, 256.000000, 0.000000))
        legend.append((14.723514, 256.000000, 0.000000))
        legend.append((29.447027, 256.000000, 0.000000))
        legend.append((44.170541, 256.000000, 0.000000))
        legend.append((58.894055, 256.000000, 0.000000))
        legend.append((73.617569, 256.000000, 0.000000))
        legend.append((88.341082, 256.000000, 0.000000))
        legend.append((103.064596, 256.000000, 0.000000))
        legend.append((117.788110, 256.000000, 0.000000))
        legend.append((132.511624, 256.000000, 0.000000))
        legend.append((147.235137, 256.000000, 0.000000))
        legend.append((161.958651, 256.000000, 0.000000))
        legend.append((176.682165, 256.000000, 0.000000))
        legend.append((188.728722, 255.665192, 0.000000))
        legend.append((192.744408, 254.325961, 0.000000))
        legend.append((196.760094, 252.986729, 0.000000))
        legend.append((200.775780, 251.647498, 0.000000))
        legend.append((204.791467, 250.308267, 0.000000))
        legend.append((208.807153, 248.969035, 0.000000))
        legend.append((212.822839, 247.629804, 0.000000))
        legend.append((216.838525, 246.290573, 0.000000))
        legend.append((220.854212, 244.951341, 0.000000))
        legend.append((224.869898, 243.612110, 0.000000))
        legend.append((228.885584, 242.272878, 0.000000))
        legend.append((232.901271, 240.933647, 0.000000))
        legend.append((236.916957, 239.594416, 0.000000))
        legend.append((239.594416, 237.586573, 0.000000))
        legend.append((240.933647, 234.910118, 0.000000))
        legend.append((242.272878, 232.233663, 0.000000))
        legend.append((243.612110, 229.557208, 0.000000))
        legend.append((244.951341, 226.880753, 0.000000))
        legend.append((246.290573, 224.204298, 0.000000))
        legend.append((247.629804, 221.527843, 0.000000))
        legend.append((248.969035, 218.851388, 0.000000))
        legend.append((250.308267, 216.174933, 0.000000))
        legend.append((251.647498, 213.498478, 0.000000))
        legend.append((252.986729, 210.822024, 0.000000))
        legend.append((254.325961, 208.145569, 0.000000))
        legend.append((255.665192, 205.469114, 0.000000))
        legend.append((256.000000, 201.788235, 0.000000))
        legend.append((256.000000, 197.772549, 0.000000))
        legend.append((256.000000, 193.756863, 0.000000))
        legend.append((256.000000, 189.741176, 0.000000))
        legend.append((256.000000, 185.725490, 0.000000))
        legend.append((256.000000, 181.709804, 0.000000))
        legend.append((256.000000, 177.694118, 0.000000))
        legend.append((256.000000, 173.678431, 0.000000))
        legend.append((256.000000, 169.662745, 0.000000))
        legend.append((256.000000, 165.647059, 0.000000))
        legend.append((256.000000, 161.631373, 0.000000))
        legend.append((256.000000, 157.615686, 0.000000))
        legend.append((256.000000, 153.600000, 0.000000))
        legend.append((256.000000, 141.552941, 0.000000))
        legend.append((256.000000, 129.505882, 0.000000))
        legend.append((256.000000, 117.458824, 0.000000))
        legend.append((256.000000, 105.411765, 0.000000))
        legend.append((256.000000, 93.364706, 0.000000))
        legend.append((256.000000, 81.317647, 0.000000))
        legend.append((256.000000, 69.270588, 0.000000))
        legend.append((256.000000, 57.223529, 0.000000))
        legend.append((256.000000, 45.176471, 0.000000))
        legend.append((256.000000, 33.129412, 0.000000))
        legend.append((256.000000, 21.082353, 0.000000))
        legend.append((256.000000, 9.035294, 0.000000))
        legend.append((255.330886, 0.000000, 0.000000))
        legend.append((252.654431, 0.000000, 0.000000))
        legend.append((249.977976, 0.000000, 0.000000))
        legend.append((247.301522, 0.000000, 0.000000))
        legend.append((244.625067, 0.000000, 0.000000))
        legend.append((241.948612, 0.000000, 0.000000))
        legend.append((239.272157, 0.000000, 0.000000))
        legend.append((236.595702, 0.000000, 0.000000))
        legend.append((233.919247, 0.000000, 0.000000))
        legend.append((231.242792, 0.000000, 0.000000))
        legend.append((228.566337, 0.000000, 0.000000))
        legend.append((225.889882, 0.000000, 0.000000))
        legend.append((223.213427, 0.000000, 0.000000))
        legend.append((221.205584, 0.000000, 0.000000))
        legend.append((219.866353, 0.000000, 0.000000))
        legend.append((218.527122, 0.000000, 0.000000))
        legend.append((217.187890, 0.000000, 0.000000))
        legend.append((215.848659, 0.000000, 0.000000))
        legend.append((214.509427, 0.000000, 0.000000))
        legend.append((213.170196, 0.000000, 0.000000))
        legend.append((211.830965, 0.000000, 0.000000))
        legend.append((210.491733, 0.000000, 0.000000))
        legend.append((209.152502, 0.000000, 0.000000))
        legend.append((207.813271, 0.000000, 0.000000))
        legend.append((206.474039, 0.000000, 0.000000))
        legend.append((205.134808, 0.000000, 0.000000))
        legend.append((204.800000, 12.047059, 12.047059))
        legend.append((204.800000, 28.109804, 28.109804))
        legend.append((204.800000, 44.172549, 44.172549))
        legend.append((204.800000, 60.235294, 60.235294))
        legend.append((204.800000, 76.298039, 76.298039))
        legend.append((204.800000, 92.360784, 92.360784))
        legend.append((204.800000, 108.423529, 108.423529))
        legend.append((204.800000, 124.486275, 124.486275))
        legend.append((204.800000, 140.549020, 140.549020))
        legend.append((204.800000, 156.611765, 156.611765))
        legend.append((204.800000, 172.674510, 172.674510))
        legend.append((204.800000, 188.737255, 188.737255))
        legend.append((204.800000, 204.800000, 204.800000))
        
        return convert_palette(legend)

    palette = []
    for i in range(256):
        palette.append((legend[i,0],
                        legend[i,1],
                        legend[i,2]))

    return palette


def convert_palette(palette):
    """Convert palette from [0,255] range to [0,1].
    """ 
    new_palette = []
    for i in palette:
        new_palette.append((i[0] / 255.0,
                            i[1] / 255.0,
                            i[2] / 255.0))
    return new_palette
