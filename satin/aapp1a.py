from ConfigParser import ConfigParser
from satin import CONFIG_PATH
import os.path

# Full scan period, i.e. the time interval between two consecutive lines (sec)
T_STEP_L = 0.1667 


def get_lat_lon(satscene, positions):
    """Get lon and lat from file.
    """    
    conf = ConfigParser()
    conf.read(os.path.join(CONFIG_PATH, satscene.fullname + ".cfg"))
    options = {}
    for option, value in conf.items(satscene.instrument_name + "-level1",
                                    raw = True):
        options[option] = value
    CASES[satscene.instrument_name](satscene, options, positions)

def get_lat_lon_avhrr(satscene, options, positions):
    """Read lat and lon.
    """

    import pyaapp
    import datetime
    
    one_minute = datetime.timedelta(minuts=1)

    t_start = satscene.time_slot
    t_end = t_start + one_minute

    epoch = datetime.datetime(1950, 1, 1)


    t50_start = (t_start - epoch)
    t50_end = (t_end - epoch)
    
    jday_start = t50_start.seconds / (3600.0 *24) + t50_start.days
    jday_end = t50_end.seconds / (3600.0 *24) + t50_end.days
    
    pyaapp.read_satpos_file(jday_start, jday_end,
                            satscene.satname+" "+satscene.number,
                            ("/data/24/saf/pps/import/ANC_data/source/satpos_"+
                             satscene.fullname+"_"+
                             satscene.time_slot.strptime("%H%M%S")+".txt"))
    
    att = pyaapp.prepare_attitude(int(satscene.number), 0, 0, 0)
    nlines = int((t_end - t_start).seconds / T_STEP_L)

    return (pyaapp.linepixel2lonlat(int(satscene.number)[1:3], 0, 0, att,
                                    jday_start, jday_end)
            for (line, col) in positions)


LAT_LON_CASES = {
    "avhrr": get_lat_lon_avhrr
    }
