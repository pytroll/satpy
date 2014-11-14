#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2014 Martin Raspaud

# Author(s):

#   Martin Raspaud <martin.raspaud@smhi.se>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Reader for eumetsat's native format.

TODO:
 - Add IMPFConfiguration Record
 - Make it an mpop reader (with geo info)
 - Make it work on a subset of channels
 - Put the hrv inside a square image (for not subsetted cases)
 - cleanup

"""
import numpy as np
from mipp.xrit.MSG import (read_proheader, read_epiheader,
                           _Calibrator, read_metadata)
from mipp.xrit.loader import ImageLoader
from mipp.xrit import Metadata

no_data_value = 0

class UImageLoader(ImageLoader):
    def __init__(self, mda, data, mask=False, calibrate=False):
        ImageLoader.__init__(self, mda, None, mask, calibrate)
        self.data = data
        
    def _read(self, lines, cols, mda):
        return self.data

def dec10to16(data):
    arr10 = data.astype(np.uint16).flat
    new_shape = list(data.shape[:-1]) + [(data.shape[-1] * 8) / 10]
    arr16 = np.zeros(new_shape, dtype=np.uint16)
    arr16.flat[::4] = np.left_shift(arr10[::5], 2) + \
        np.right_shift((arr10[1::5]), 6)
    arr16.flat[1::4] = np.left_shift((arr10[1::5] & 63), 4) + \
        np.right_shift((arr10[2::5]), 4)
    arr16.flat[2::4] = np.left_shift(arr10[2::5] & 15, 6) + \
        np.right_shift((arr10[3::5]), 2)
    arr16.flat[3::4] = np.left_shift(arr10[3::5] & 3, 8) + \
        arr10[4::5]
    return arr16   

def show(data):
    """Show the stetched data.
    """
    import Image as pil
    img = pil.fromarray(np.array((data - data.min()) * 255.0 /
                                 (data.max() - data.min()), np.uint8))
    img.show()

CHANNELS = {"VIS006": 1,
            "VIS008": 2,
            "IR_016": 3,
            "IR_039": 4,
            "WV_062": 5,
            "WV_073": 6,
            "IR_087": 7,
            "IR_097": 8,
            "IR_108": 9,
            "IR_120": 10,
            "IR_134": 11,
            "HRV": 12}

def _get_metadata(hdr, ftr, channel_name):
    
    md = Metadata()
    md.calibrate = _Calibrator(hdr, channel_name)

    if channel_name == "HRV":
        md.pixel_size = (1000.134348869, 1000.134348869)
    else:
        md.pixel_size = (3000.403165817, 3000.403165817)

    md.sublon = hdr["ProjectionDescription"]["LongitudeOfSSP"]
    md.channel = channel_name
    if md.channel == "HRV":
        md.image_size = np.array((hdr["ReferenceGridHRV"]["NumberOfLines"],
                                  hdr["ReferenceGridHRV"]["NumberOfColumns"]))
    else:
        md.image_size = np.array((hdr["ReferenceGridVIS_IR"]["NumberOfLines"],
                                  hdr["ReferenceGridVIS_IR"]["NumberOfColumns"]))
        
    #md.satname = im.platform.lower()
    md.product_type = 'full disc'
    md.region_name = 'full disc'
    if md.channel == "HRV":
        md.first_pixel = hdr["ReferenceGridHRV"]["GridOrigin"]
        ns_, ew_ = md.first_pixel.split()
        del ns_
        md.boundaries = np.array([[
            ftr["LowerSouthLineActual"],
            ftr["LowerNorthLineActual"],
            ftr["LowerEastColumnActual"],
            ftr["LowerWestColumnActual"]],
           [ftr["UpperSouthLineActual"],
            ftr["UpperNorthLineActual"],
            ftr["UpperEastColumnActual"],
            ftr["UpperWestColumnActual"]]])

        hcoff = 1856 * 3
        hloff = 1856 * 3
        md.coff = (ftr["Lower"+ew_.capitalize()+"ColumnActual"]
                   + hcoff - 1)
        md.loff = hloff

    else:
        md.first_pixel = hdr["ReferenceGridVIS_IR"]["GridOrigin"]
        ns_, ew_ = md.first_pixel.split()
        md.boundaries = np.array([[
            ftr["SouthernLineActual"],
            ftr["NorthernLineActual"],
            ftr["EasternColumnActual"],
            ftr["WesternColumnActual"]]])
        lcoff = 1856
        lloff = 1856
        md.coff = lcoff
        md.loff = lloff

    md.no_data_value = no_data_value
    md.line_offset = 0
    #md.time_stamp = im.time_stamp
    #md.production_time = im.production_time
    md.calibration_unit = 'counts'

    return md

def load(satscene, calibrate=1):
    test_file = "/local_disk/data/satellite/umarf/MSG3-SEVI-MSG15-0100-NA-20131109121244.570000000Z-1080742.nat"
    _load_from_file(test_file, satscene, calibrate)

def _load_from_file(filename, satscene, calibrate):
    hdr, ftr, umarf, data = linear_load(filename)
    for channel in satscene.channels_to_load:
        mda = _get_metadata(hdr, ftr, channel)
        if channel == "HRV":
            dat = dec10to16(data["hrv"]["line_data"]).reshape((int(umarf["NumberLinesHRV"]))) * 1.0
        else:
            dat = dec10to16(data["visir"]["line_data"][:, CHANNELS[channel], :]) * 1.0
            print dat.min(), dat.max()
        uil = UImageLoader(mda, dat, mask=False, calibrate=calibrate)
        md, res = uil()

        satscene[channel] = np.ma.masked_equal(res, 0)
        proj_params = 'proj=geos lon_0=9.50 lat_0=0.00 a=6378169.00 b=6356583.80 h=35785831.00'.split()
        proj_dict = {}
        for param in proj_params:
            key, val = param.split("=")
            proj_dict[key] = val
        from pyresample import geometry
        satscene[channel].area = geometry.AreaDefinition(
            satscene.satname + satscene.instrument_name +
            str(md.area_extent) +
                str(res.shape),
                "On-the-fly area",
                proj_dict["proj"],
                proj_dict,
                res.shape[1],
                res.shape[0],
                md.area_extent)
        


def linear_load(filename):
    """First draft, works to retrieve counts.
    """
    with open(filename) as fp_:
        umarf = {}
        for i in range(6):
            name = (fp_.read(30).strip("\x00"))[:-2].strip()
            umarf[name] = fp_.read(50).strip("\x00").strip()

        for i in range(27):
            name = fp_.read(30).strip("\x00")
            if name == '':
                fp_.read(32)
                continue
            name = name[:-2].strip()
            umarf[name] = {"size": fp_.read(16).strip("\x00").strip(),
                           "adress": fp_.read(16).strip("\x00").strip()}
        for i in range(19):
            name = (fp_.read(30).strip("\x00"))[:-2].strip()
            umarf[name] = fp_.read(50).strip("\x00").strip()

        for i in range(18):
            name = (fp_.read(30).strip("\x00"))[:-2].strip()
            umarf[name] = fp_.read(50).strip("\x00").strip()

        from pprint import pprint
        pprint(umarf)
        uhdrlen = fp_.tell()
        print "UMARF header length", uhdrlen


        gp_pk_header = np.dtype([
            ("HeaderVersionNo", ">i1"),
            ("PacketType", ">i1"),
            ("SubHeaderType", ">i1"),
            ("SourceFacilityId", ">i1"),
            ("SourceEnvId", ">i1"),
            ("SourceInstanceId", ">i1"),
            ("SourceSUId", ">i4"),
            ("SourceCPUId", ">i1", (4, )),
            ("DestFacilityId", ">i1"),
            ("DestEnvId", ">i1"),
            ("SequenceCount", ">u2"),
            ("PacketLength", ">i4"),
            ])

        gp_pk_subheader = np.dtype([
            ("SubHeaderVersionNo", ">i1"),
            ("ChecksumFlag", ">i1"),
            ("Acknowledgement", ">i1", (4, )),
            ("ServiceType", ">i1"),
            ("ServiceSubtype", ">i1"),
            ("PacketTime", ">i1", (6, )),
            ("SpacecraftId", ">i2"),
            ])

        pk_head = np.dtype([("gp_pk_header", gp_pk_header),
                            ("gp_pk_sh1", gp_pk_subheader)])

        # read header

        pk_header = np.fromfile(fp_, pk_head, count=1)

        hdr_version = ord(fp_.read(1))
        hdr = read_proheader(fp_)

        # skip IMPF CONFIGURATION
        fp_.seek(19786, 1)


        # read line data

        cols_visir = np.ceil(int(umarf["NumberColumnsVISIR"]) * 5.0 / 4) # 4640
        if (int(umarf['WestColumnSelectedRectangle'])
            - int(umarf['EastColumnSelectedRectangle'])) < 3711:
            cols_hrv = np.ceil(int(umarf["NumberColumnsHRV"]) * 5.0 / 4) # 6960
        else:
            cols_hrv = np.ceil(5568 * 5.0 / 4) # 6960


        #fp_.seek(450400)

        # FIXME: works only if all channels are selected!
        selected = umarf["SelectedBandIDs"]
        visir_nb = selected.count("X", 0, 11)
        hrv_nb = selected.count("X", 11, 12)

        print visir_nb, hrv_nb

        visir_type = np.dtype([("gp_pk", pk_head),
                               ("version", ">u1"),
                               ("satid", ">u2"),
                               ("time", ">u2", (5, )),
                               ("lineno", ">u4"),
                               ("chan_id", ">u1"),
                               ("acq_time", ">u2", (3, )),
                               ("line_validity", ">u1"),
                               ("line_rquality", ">u1"),
                               ("line_gquality", ">u1"),
                               ("line_data", ">u1", (cols_visir, ))])

        hrv_type = np.dtype([("gp_pk", pk_head),
                             ("version", ">u1"),
                             ("satid", ">u2"),
                             ("time", ">u2", (5, )),
                             ("lineno", ">u4"),
                             ("chan_id", ">u1"),
                             ("acq_time", ">u2", (3, )),
                             ("line_validity", ">u1"),
                             ("line_rquality", ">u1"),
                             ("line_gquality", ">u1"),
                             ("line_data", ">u1", (cols_hrv, ))])

        if hrv_nb == 0:
            linetype = np.dtype([("visir", visir_type, (visir_nb, ))])
        elif visir_nb == 0:
            linetype = np.dtype([("hrv", hrv_type, (hrv_nb * 3, ))])
        else:
            linetype = np.dtype([("visir", visir_type, (visir_nb, )),
                                 ("hrv", hrv_type, (hrv_nb * 3, ))])


        data_len = int(umarf["NumberLinesVISIR"])

        # read everything in memory
        #res = np.fromfile(fp_, dtype=linetype, count=data_len)

        # lazy reading
        res = np.memmap(fp_, dtype=linetype, shape=(data_len, ), offset=450400, mode="r")

        fp_.seek(linetype.itemsize * data_len + 450400)

        # read trailer
        pk_header = np.fromfile(fp_, pk_head, count=1)
        ftr = read_epiheader(fp_)


        return hdr, ftr, umarf, res


if __name__ == '__main__':

    test_file = "/local_disk/data/satellite/umarf/MSG3-SEVI-MSG15-0100-NA-20131109121244.570000000Z-1080742.nat"
    #test_file = sys.argv[1]
    hdr, ftr, umarf, res = linear_load(test_file)
    
    # display the data
    show(dec10to16(res["visir"]["line_data"][:, 1, :])[::-1, ::-1])
    #show(dec10to16(res["hrv"]["line_data"]).reshape((int(umarf["NumberLinesHRV"]), -1))[::-1, ::-1])
