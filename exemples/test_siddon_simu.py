#!/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
# data 
image_header = {'n_images':16,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':256, 'NAXIS2':256, 
                'CRPIX1':128, 'CRPIX2':128,
                'CDELT1':6e-5, 'CDELT2':6e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
data = siddon.simu.circular_trajectory_data(**image_header)
data[:] = np.zeros(data.shape)
# object
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':64, 'NAXIS2':64, 'NAXIS3':64,
          'CRPIX1':32., 'CRPIX2':32., 'CRPIX3':32.,
          'CDELT1':0.04, 'CDELT2':0.04, 'CDELT3':0.04,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
header['radius'] = 1.2
obj = siddon.simu.spherical_object(**header)
# projector
P = siddon.siddon_lo(data.header, obj.header)
# projection
t = time.time()
data = siddon.projector(data, obj)
print("projection time : " + str(time.time() - t))
# backprojection
data1 = data.copy()
data1[:] = 0.
data1[64:192, 64:192] = 1.
obj0 = obj.copy()
obj0[:] = 0.
t = time.time()
bpj = siddon.backprojector(data1, obj0)
print("projection time : " + str(time.time() - t))
# inversion using scipy.sparse.linalg
t = time.time()
M = P.T * P
b = P.T * data.flatten()
sol, info = spl.bicgstab(M, b, maxiter=100, tol=1e-2)
sol = sol.reshape(bpj.shape)
if info != 0:
    print("Inversion algorithm did not converge to " + str(tol))
print("inversion time : " + str(time.time() - t))
