#!/usr/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
import lo
# object
header = {'BITPIX':-64,
          'NAXIS1':64, 'NAXIS2':64, 'NAXIS3':64,
          'CRPIX1':32., 'CRPIX2':32., 'CRPIX3':32.,
          'CDELT1':0.16, 'CDELT2':0.16, 'CDELT3':0.16,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
header['radius'] = 1.2
obj = siddon.simu.object_from_header(header)
obj[:] = siddon.phantom.shepp_logan(obj.shape)
# data 
image_header = {'n_images':60,
                'BITPIX':-64,
                'NAXIS1':128, 'NAXIS2':128,
                'CRPIX1':64., 'CRPIX2':64.,
                'CDELT1':48e-5, 'CDELT2':48e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
image_header['max_lon'] = np.pi
data = siddon.simu.circular_trajectory_data(**image_header)
data[:] = np.zeros(data.shape)
# model
kwargs = {"pb":"pb", "obj_rmin":1.5, "data_rmin":1.5}
P, D, obj_mask, data_mask = siddon.models.thomson(data, obj, u=.5, **kwargs)
# projection
t = time.time()
data = (P * obj.flatten()).reshape(data.shape)
print("projection time : " + str(time.time() - t))
# data
y = data.flatten()
# backprojection
t = time.time()
x0 = P.T * y
bpj = x0.reshape(obj.shape)
print("backprojection time : " + str(time.time() - t))
# coverage map
weights = (P.T * np.ones(y.size)).reshape(obj.shape)
# hyperparameters
hypers = 1e-2 * np.ones(3)
#Ds, hypers = [], []
# inversion using scipy.sparse.linalg
t = time.time()
tol = 1e-8
sol = lo.acg(P, y, D, hypers,  maxiter=100, tol=tol)
sol = sol.reshape(bpj.shape)
print("inversion time : " + str(time.time() - t))
