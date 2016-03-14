# -*- coding: utf-8 -*-
import numpy as np
from pyhdf.SD import *
from argparse import ArgumentParser
from satutil_lib import *
import hdf_output as ho
import os


bUseV171 = False

if bUseV171:
    UO3 = 0.319
    UH2O = 2.93
else:
    UO3 = 0.285
    UH2O = 2.93

MAXSOLZ = 86.5
MAXAIRMASS = 18
SCALEHEIGHT = 8000
FILL_INT16 =32767

TAUSTEP4SPHALB = 0.0001
MAXNUMSPHALBVALUES = 4000    # with no aerosol taur <= 0.4 in all bands everywhere

def csalbr(tau):
    # Previously 3 functions csalbr fintexp1, fintexp3
    a= [ -.57721566, 0.99999193, -0.24991055, 0.05519968, -0.00976004, 0.00107857]
    xx = a[0] + a[1]*tau + a[2]*tau**2 + a[3]*tau**3 + a[4]*tau**4 + a[5]*tau**5

    # xx = a[0]
    # xftau = 1.0
    # for i in xrange(5):
    #     xftau = xftau*tau
    #     xx = xx + a[i] * xftau
    fintexp1 = xx-np.log(tau)
    fintexp3 = (np.exp(-tau) * (1.0 - tau) + tau**2 * fintexp1) / 2.0

    return (3.0 * tau - fintexp3 * (4.0 + 2.0 * tau) + 2.0 * np.exp(-tau)) / (4.0 + 3.0 * tau)

# From crefl.1.7.1
if bUseV171:
    aH2O =  np.array([-5.60723, -5.25251, 0, 0, -6.29824, -7.70944, -3.91877, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    bH2O =  np.array([0.820175, 0.725159, 0, 0, 0.865732, 0.966947, 0.745342, 0, 0, 0, 0, 0, 0, 0, 0, 0 ])
    #const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0, 0, 0, 0.00244, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263};*/
    aO3 =  np.array([0.0715289, 0, 0.00743232, 0.089691, 0, 0, 0, 0.001, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263])
    #const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,  0.0036,  0.0012,  0.0004,  0.3109, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array([0.05100, 0.01631, 0.19325, 0.09536, 0.00366, 0.00123, 0.00043, 0.3139, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155])
else:
    #From polar2grid cviirs.c
    aH2O =  np.array([0.000406601 ,0.0015933 , 0,1.78644e-05 ,0.00296457 ,0.000617252 , 0.000996563,0.00222253 ,0.00094005 , 0.000563288, 0, 0, 0, 0, 0, 0 ])
    bH2O =  np.array([0.812659,0.832931 , 1., 0.8677850, 0.806816 , 0.944958 ,0.78812 ,0.791204 ,0.900564 ,0.942907 , 0, 0, 0, 0, 0, 0 ])
    #/*const float aO3[Nbands]={ 0.0711,    0.00313, 0.0104,     0.0930,   0, 0, 0, 0.00244, 0.00383, 0.0225, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263};*/
    aO3 =  np.array([ 0.0433461, 0.0,    0.0178299   ,0.0853012 , 0, 0, 0, 0.0813531,   0, 0, 0.0663, 0.0836, 0.0485, 0.0395, 0.0119, 0.00263])
    #/*const float taur0[Nbands] = { 0.0507,  0.0164,  0.1915,  0.0948,  0.0036,  0.0012,  0.0004,  0.3109, 0.2375, 0.1596, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155};*/
    taur0 = np.array([0.04350, 0.01582, 0.16176, 0.09740,0.00369 ,0.00132 ,0.00033 ,0.05373 ,0.01561 ,0.00129, 0.1131, 0.0994, 0.0446, 0.0416, 0.0286, 0.0155])


def crefl_viirs_iff(output_path=None,IFF_fname=None,terrain=None):
    if not IFF_fname and not terrain:
        print 'crefl_viirs_iff: ERROR no input file given'
        sys.exit(-1)

    (adir,fname) = os.path.split(IFF_fname)
    xfname = fname.replace('IFFSDR','IFFCREFL')
    if not output_path:
        output_path = adir

    output_file_name = os.path.join(output_path,xfname)
    print 'crefl_viirs_iif.py: output file name: %s' % (output_file_name)

    sds_names = ['ReflectiveBandCenters','ReflectiveSolarBands','SensorAzimuth','SensorZenith','SolarAzimuth','SolarZenith']

    fp = SD(terrain)
    dem = fp.select('averaged elevation')[:]
    fp.end()

    fp = SD(IFF_fname)

    lat = fp.select('Latitude')[:]
    lon = fp.select('Longitude')[:]

    # Get digital elevation map data for our granule, set ocean fill value to 0
    #ipdb.set_trace()
    row = np.int32((90.0 - lat) *  np.shape(dem)[0]/ 180.0);
    col = np.int32((lon + 180.0) * np.shape(dem)[1]/ 360.0);
    height=np.float64(dem[row,col])
    ii,jj = np.where(np.less(height,0.0))
    height[ii,jj] = 0.0

    del lat, lon, row, col, ii, jj

    SensorAzimuth = fp.select('SensorAzimuth')[:]
    SensorZenith = fp.select('SensorZenith')[:]

    SolarAzimuth = fp.select('SolarAzimuth')[:]
    SolarZenith = fp.select('SolarZenith')[:]

    DEG2RAD = np.pi/180.0
    mus = np.cos(SolarZenith * DEG2RAD)
    muv = np.cos(SensorZenith * DEG2RAD)
    phi = SolarAzimuth - SensorAzimuth

    del SolarAzimuth, SolarZenith, SensorZenith, SensorAzimuth

    # From GetAtmVariables
    tau_step = np.linspace(TAUSTEP4SPHALB, MAXNUMSPHALBVALUES*TAUSTEP4SPHALB, MAXNUMSPHALBVALUES)
    #ipdb.set_trace()
    sphalb0 = csalbr(tau_step);

    air_mass = 1.0/mus + 1/muv;
    ii,jj = np.where(np.greater(air_mass,MAXAIRMASS))
    air_mass[ii,jj] = -1.0

    # FROM FUNCTION CHAND
    # phi: azimuthal difference between sun and observation in degree
    #      (phi=0 in backscattering direction)
    # mus: cosine of the sun zenith angle
    # muv: cosine of the observation zenith angle
    # taur: molecular optical depth
    # rhoray: molecular path reflectance
    # constant xdep: depolarization factor (0.0279)
    #          xfd = (1-xdep/(2-xdep)) / (1 + 2*xdep/(2-xdep)) = 2 * (1 - xdep) / (2 + xdep) = 0.958725775
    # */
    xfd = 0.958725775;
    xbeta2 = 0.5
    #         float pl[5];
    #         double fs01, fs02, fs0, fs1, fs2;
    as0 = [0.33243832, 0.16285370, -0.30924818, -0.10324388, 0.11493334,
        -6.777104e-02, 1.577425e-03, -1.240906e-02, 3.241678e-02, -3.503695e-02]
    as1 = [0.19666292, -5.439061e-02]
    as2 = [0.14545937, -2.910845e-02]
    #         float phios, xcos1, xcos2, xcos3;
    #         float xph1, xph2, xph3, xitm1, xitm2;
    #         float xlntaur, xitot1, xitot2, xitot3;
    #         int i, ib;

    phios = phi + 180.0
    xcos1 = 1.0
    xcos2 = np.cos(phios * DEG2RAD)
    xcos3 = np.cos(2.0 * phios * DEG2RAD)
    xph1 = 1.0 + (3.0 * mus * mus - 1.0) * (3.0 * muv * muv - 1.0) * xfd / 8.0
    xph2 = -xfd * xbeta2 * 1.5 * mus * muv * np.sqrt(1.0 - mus * mus) * np.sqrt(1.0 - muv * muv)
    xph3 = xfd * xbeta2 * 0.375 * (1.0 - mus * mus) * (1.0 - muv * muv)

    # pl[0] = 1.0
    # pl[1] = mus + muv
    # pl[2] = mus * muv
    # pl[3] = mus * mus + muv * muv
    # pl[4] = mus * mus * muv * muv

    fs01 = as0[0] + (mus + muv)*as0[1] + (mus * muv)*as0[2] + (mus * mus + muv * muv)*as0[3] + (mus * mus * muv * muv)*as0[4]
    fs02 = as0[5] + (mus + muv)*as0[6] + (mus * muv)*as0[7] + (mus * mus + muv * muv)*as0[8] + (mus * mus * muv * muv)*as0[9]
    #         for (i = 0; i < 5; i++) {
    #                 fs01 += (double) (pl[i] * as0[i]);
    #                 fs02 += (double) (pl[i] * as0[5 + i]);
    #         }

    v2m=[1,#M7 */  /* MODIS BAND 2 */ /* 865 nm  */ /* 856.5  nm */
         2,#M3 */  /* MODIS BAND 3 */ /* 488 nm  */ /* 465.6  nm */
         3,#M4 */  /* MODIS BAND 4 */ /* 555 nm  */ /* 553.6  nm */
         4,#M8 */  /* MODIS BAND 5 */ /* 1024 nm */ /* 1241.6 nm */
         5,#M10 */ /* MODIS BAND 6 */ /* 1610 nm */ /* 1629.1 nm */
         6,#M11 */ /* MODIS BAND 7 */ /* 2250 nm */ /* 2114.1 nm */
         0,#M5  */ /* MODIS BAND 1 */ /* 672  nm */ /* 645.5  nm */
         2]#M2 */  /* MODIS BAND 3 */ /* 445  nm */ /* 465.6  nm */
         # M2, M3, M4, M5, M6, M8, M7, M10, M11

    v2m_str=['M7',
         'M3',
         'M4',
         'M8',
         'M10',
         'M11',
         'M5',
         'M2']

    iff2v = [7, 3, 4, 8, 10, 11, 5, 2]
    for i in range(len(iff2v)):
        iff2v[i] -= 1

    Nbands = [0, 1, 2, 3, 4, 5, 6, 7]

    print "Processing band:"
    odata = []
    for i in Nbands:
        ib = v2m[i]
        print (i, ib, iff2v[i])
        taur = taur0[ib] * np.exp(-height / SCALEHEIGHT);
        xlntaur = np.log(taur)
        fs0 = fs01 + fs02 * xlntaur
        fs1 = as1[0] + xlntaur * as1[1]
        fs2 = as2[0] + xlntaur * as2[1]
        del xlntaur
        trdown = np.exp(-taur / mus)
        trup= np.exp(-taur / muv)
        xitm1 = (1.0 - trdown * trup) / 4.0 / (mus + muv)
        xitm2 = (1.0 - trdown) * (1.0 - trup)
        xitot1 = xph1 * (xitm1 + xitm2 * fs0)
        xitot2 = xph2 * (xitm1 + xitm2 * fs1)
        xitot3 = xph3 * (xitm1 + xitm2 * fs2)
        rhoray = xitot1 * xcos1 + xitot2 * xcos2 * 2.0 + xitot3 * xcos3 * 2.0

        sphalb = sphalb0[np.int32(taur / TAUSTEP4SPHALB + 0.5)]
        Ttotrayu = ((2 / 3. + muv) + (2 / 3. - muv) * trup) / (4 / 3. + taur)
        Ttotrayd = ((2 / 3. + mus) + (2 / 3. - mus) * trdown) / (4 / 3. + taur)
        tO3 = 1.0
        tO2 = 1.0
        tH2O = 1.0

        if aO3[ib] != 0:
            tO3 = np.exp(-air_mass * UO3 * aO3[ib])
        if bH2O[ib] != 0:
            if bUseV171:
                tH2O = np.exp(-np.exp(aH2O[ib] + bH2O[ib] * log(air_mass * UH2O)))
            else:
                tH2O = np.exp(-(aH2O[ib]*(np.power((air_mass * UH2O),bH2O[ib]))))
        #t02 = exp(-m * aO2)
        TtotraytH2O = Ttotrayu * Ttotrayd * tH2O
        tOG = tO3 * tO2

        # float correctedrefl(float refl, float TtotraytH2O, float tOG, float rhoray, float sphalb)
        refl = np.squeeze(fp.select('ReflectiveSolarBands')[iff2v[i],:,:])
        cent = np.squeeze(fp.select('ReflectiveBandCenters')[iff2v[i]])
        print cent

    	ii = np.where(np.greater_equal(refl,65528))
        corr_refl = (refl / tOG - rhoray) / TtotraytH2O
        corr_refl = corr_refl/(1.0 + corr_refl * sphalb);
        data_type = 'float32'
        corr_refl[ii] = np.float32(-999.00)
        odata.append(ho.OutData(corr_refl, 'Corr_Refl_'+v2m_str[i], data_type, np.shape(corr_refl)))

        # IFFSDR_npp_d20140111_t185000_c20140213044424_ssec_dev.hdf

    ho.write_hdf(output_file_name, odata, 'hdf4')

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    #group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('-o', '--output_path')
    #parser.add_argument('-iffcth','--IFF-CTH', action='store_true', default=False, help='VIIRS IFF CTH imagery')
    parser.add_argument('-f','--IFF-fname', help='IFF file name', required=True)
    parser.add_argument('-t','--terrain', help='terrain file name', required=True)
    args = parser.parse_args()
    crefl_viirs_iff(output_path=args.output_path,IFF_fname=args.IFF_fname,terrain=args.terrain)
