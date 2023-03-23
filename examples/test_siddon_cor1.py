#!/usr/bin/env python
import numpy as np
import os
import copy
import time
import tomograpy
import fitsarray as fa
import lo

# data 
path = os.path.join(os.getenv('HOME'), 'data', 'tomograpy., 'cor1')
#obsrvtry = 'SOHO    '
#instrume = 'LASCO   '
time_window = ['2009/09/01 00:00:00.000', '2009/09/15 00:00:00.000']
time_step = 8 * 3600. # one image every time_step seconds
data = tomograpy.solar.read_data(path, bin_factor=8,
                               #time_window=time_window, 
                               #time_step=time_step
                               )
# errors in data ...
data.header['BITPIX'][:] = -64
data[np.isnan(data)] = 0.
data.header['RSUN'] /= 16.
# cube
shape = np.asarray(3 * (128. ,))
crpix = shape / 2.
cdelt = 6. / shape
crval = np.zeros(3)
header = {'CRPIX1':crpix[0], 'CRPIX2':crpix[1], 'CRPIX3':crpix[2],
          'CDELT1':cdelt[0], 'CDELT2':cdelt[1], 'CDELT3':cdelt[2],
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
cube = fa.zeros(shape, header=header)
t = time.time()
cube = tomograpy.backprojector(data, cube, obstacle="sun")
print("backprojection time : " + str(time.time() - t))

# inversion
t = time.time()
u = .5
kwargs={
    "obj_rmin":1.5,
    "obj_rmax":3.,
    "data_rmin":1.5,
    "data_rmax":2.5,
    "mask_negative":True
}
P, D, obj_mask, data_mask = tomograpy.models.thomson(data, cube, u, **kwargs)
# bpj
b = data.flatten()
bpj = (P.T * b).reshape(cube.shape)
hypers = 1e3 * np.ones(3)
sol = lo.acg(P, b, D, hypers, maxiter=100, tol=1e-6)
print("inversion time : %f" % (time.time() - t))
# reshape solution
sol.resize(cube.shape)
sol = fa.asfitsarray(sol, header=cube.header)
# reproject solution
reproj = P * sol.ravel()
reproj.resize(data.shape)
