import numpy as np

def modis2spec(modis):
    z = np.zeros_like(modis["1"].data)
    spec = [z, z, z, z, z,
            modis["8"].data, modis["8"].data,
            modis["8"].data, modis["8"].data,
            z, z, z,
            modis["9"].data, modis["9"].data,
            modis["9"].data,
            z,
            modis["3"].data, modis["3"].data,
            modis["3"].data, modis["3"].data,
            modis["3"].data, 
            modis["10"].data, modis["10"].data,
            modis["10"].data,
            z, z, z, z, z,
            modis["11"].data, modis["11"].data,
            modis["11"].data,
            z,
            modis["12"].data, modis["12"].data,
            modis["12"].data,
            modis["4"].data, modis["4"].data,
            z, z, z, z, z,
            z, z, z, z, z,
            modis["1"].data, modis["1"].data,
            modis["1"].data, modis["1"].data,
            modis["1"].data, modis["1"].data,
            modis["1"].data, modis["1"].data,
            modis["13lo"].data, modis["13lo"].data,
            modis["13lo"].data,
            modis["14lo"].data, modis["14lo"].data,
            modis["14lo"].data,
            z, z, z, z, z,
            z, z, z, z, z,
            z,
            modis["15"].data, modis["15"].data,
            modis["15"].data,
            z, z, z, z, z]
    return spec
                     

def spec2rgb(spec):
    """Spec should be a list from 0.380 to 0.780 um, with 0.005 um steps, of
    pixel arrays.
    """

    cie_colour_match = np.array(
        [(0.0014,0.0000,0.0065), (0.0022,0.0001,0.0105), (0.0042,0.0001,0.0201),
         (0.0076,0.0002,0.0362), (0.0143,0.0004,0.0679), (0.0232,0.0006,0.1102),
         (0.0435,0.0012,0.2074), (0.0776,0.0022,0.3713), (0.1344,0.0040,0.6456),
         (0.2148,0.0073,1.0391), (0.2839,0.0116,1.3856), (0.3285,0.0168,1.6230),
         (0.3483,0.0230,1.7471), (0.3481,0.0298,1.7826), (0.3362,0.0380,1.7721),
         (0.3187,0.0480,1.7441), (0.2908,0.0600,1.6692), (0.2511,0.0739,1.5281),
         (0.1954,0.0910,1.2876), (0.1421,0.1126,1.0419), (0.0956,0.1390,0.8130),
         (0.0580,0.1693,0.6162), (0.0320,0.2080,0.4652), (0.0147,0.2586,0.3533),
         (0.0049,0.3230,0.2720), (0.0024,0.4073,0.2123), (0.0093,0.5030,0.1582),
         (0.0291,0.6082,0.1117), (0.0633,0.7100,0.0782), (0.1096,0.7932,0.0573),
         (0.1655,0.8620,0.0422), (0.2257,0.9149,0.0298), (0.2904,0.9540,0.0203),
         (0.3597,0.9803,0.0134), (0.4334,0.9950,0.0087), (0.5121,1.0000,0.0057),
         (0.5945,0.9950,0.0039), (0.6784,0.9786,0.0027), (0.7621,0.9520,0.0021),
         (0.8425,0.9154,0.0018), (0.9163,0.8700,0.0017), (0.9786,0.8163,0.0014),
         (1.0263,0.7570,0.0011), (1.0567,0.6949,0.0010), (1.0622,0.6310,0.0008),
         (1.0456,0.5668,0.0006), (1.0026,0.5030,0.0003), (0.9384,0.4412,0.0002),
         (0.8544,0.3810,0.0002), (0.7514,0.3210,0.0001), (0.6424,0.2650,0.0000),
         (0.5419,0.2170,0.0000), (0.4479,0.1750,0.0000), (0.3608,0.1382,0.0000),
         (0.2835,0.1070,0.0000), (0.2187,0.0816,0.0000), (0.1649,0.0610,0.0000),
         (0.1212,0.0446,0.0000), (0.0874,0.0320,0.0000), (0.0636,0.0232,0.0000),
         (0.0468,0.0170,0.0000), (0.0329,0.0119,0.0000), (0.0227,0.0082,0.0000),
         (0.0158,0.0057,0.0000), (0.0114,0.0041,0.0000), (0.0081,0.0029,0.0000),
         (0.0058,0.0021,0.0000), (0.0041,0.0015,0.0000), (0.0029,0.0010,0.0000),
         (0.0020,0.0007,0.0000), (0.0014,0.0005,0.0000), (0.0010,0.0004,0.0000),
         (0.0007,0.0002,0.0000), (0.0005,0.0002,0.0000), (0.0003,0.0001,0.0000),
         (0.0002,0.0001,0.0000), (0.0002,0.0001,0.0000), (0.0001,0.0000,0.0000),
         (0.0001,0.0000,0.0000), (0.0001,0.0000,0.0000), (0.0000,0.0000,0.0000)])


    # this is too memory expensive...
    #ciex =  np.repeat(cie_colour_match[:, 0], spec.shape[1] * spec.shape[2]).reshape((spec.shape[0], spec.shape[1], spec.shape[2]))
    #ciey =  np.repeat(cie_colour_match[:, 1], spec.shape[1] * spec.shape[2]).reshape((spec.shape[0], spec.shape[1], spec.shape[2]))
    #ciez =  np.repeat(cie_colour_match[:, 2], spec.shape[1] * spec.shape[2]).reshape((spec.shape[0], spec.shape[1], spec.shape[2]))

    #x = np.sum(ciex * spec)
    #y = np.sum(ciey * spec)
    #z = np.sum(ciez * spec)

    x = np.zeros((spec[0].shape[0],spec[0].shape[1]))
    y = np.zeros((spec[0].shape[0],spec[0].shape[1]))
    z = np.zeros((spec[0].shape[0],spec[0].shape[1]))

    for i in range(len(spec)):
        x = x + cie_colour_match[i, 0] * spec[i]
        y = y + cie_colour_match[i, 1] * spec[i]
        z = z + cie_colour_match[i, 2] * spec[i]
        


    
    xyz = x + y + z
    xc = x #/ xyz 
    yc = y #/ xyz 
    zc = z #/ xyz 
    
    #SMPTEsystem =  { "SMPTE",              0.630,  0.340,  0.310,  0.595,  0.155,  0.070,  IlluminantD65,  GAMMA_REC709 },

    xRed = 0.630
    yRed = 0.340
    xGreen = 0.310
    yGreen = 0.595
    xBlue = 0.155
    yBlue = 0.070
    xWhite = 0.3127
    yWhite = 0.3291

    xr = xRed
    yr = yRed
    zr = 1 - (xr + yr)
    xg = xGreen
    yg = yGreen
    zg = 1 - (xg + yg)
    xb = xBlue
    yb = yBlue
    zb = 1 - (xb + yb)

    xw = xWhite
    yw = yWhite
    zw = 1 - (xw + yw)

    #/* xyz -> rgb matrix, before scaling to white. */
    
    rx = (yg * zb) - (yb * zg)
    ry = (xb * zg) - (xg * zb)
    rz = (xg * yb) - (xb * yg)
    gx = (yb * zr) - (yr * zb)
    gy = (xr * zb) - (xb * zr)
    gz = (xb * yr) - (xr * yb)
    bx = (yr * zg) - (yg * zr)
    by = (xg * zr) - (xr * zg)
    bz = (xr * yg) - (xg * yr)

    #/* White scaling factors.
    #   Dividing by yw scales the white luminance to unity, as conventional. */
       
    rw = ((rx * xw) + (ry * yw) + (rz * zw)) / yw
    gw = ((gx * xw) + (gy * yw) + (gz * zw)) / yw
    bw = ((bx * xw) + (by * yw) + (bz * zw)) / yw

    #/* xyz -> rgb matrix, correctly scaled to white. */
    
    rx = rx / rw
    ry = ry / rw
    rz = rz / rw
    gx = gx / gw
    gy = gy / gw
    gz = gz / gw
    bx = bx / bw
    by = by / bw
    bz = bz / bw

    #/* rgb of the desired point */
    
    r = (rx * xc) + (ry * yc) + (rz * zc)
    g = (gx * xc) + (gy * yc) + (gz * zc)
    b = (bx * xc) + (by * yc) + (bz * zc)
    
    return (r, g, b)
