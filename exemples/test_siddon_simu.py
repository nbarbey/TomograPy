#!/usr/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
import lo
# object
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':32, 'NAXIS2':32, 'NAXIS3':32,
          'CRPIX1':16., 'CRPIX2':16., 'CRPIX3':16.,
          'CDELT1':0.04, 'CDELT2':0.04, 'CDELT3':0.04,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
header['radius'] = 1.2
obj = siddon.simu.object_from_header(header)
obj[:] = siddon.phantom.shepp_logan(obj.shape)
# data 
image_header = {'n_images':60,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':128, 'NAXIS2':128,
                'CRPIX1':64, 'CRPIX2':64,
                'CDELT1':6e-5, 'CDELT2':6e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
data = siddon.simu.circular_trajectory_data(**image_header)
data[:] = np.zeros(data.shape)
# projector
P = siddon.siddon_lo(data.header, obj.header)
# projection
t = time.time()
data = siddon.projector(data, obj)
print("projection time : " + str(time.time() - t))
# data
y = data.flatten()
# backprojection
t = time.time()
x0 = P.T * y
bpj = x0.reshape(obj.shape)
print("projection time : " + str(time.time() - t))
# coverage map
weights = (P.T * np.ones(y.size)).reshape(obj.shape)
# priors
Ds = [lo.diff(obj.shape, axis=i) for i in xrange(3)]
hypers = 1e0 * np.ones(3)
#Ds, hypers = [], []
# inversion using scipy.sparse.linalg
t = time.time()
tol = 1e-5
sol, info = lo.rls(P, y, Ds, hypers,  maxiter=100, tol=tol)
sol = sol.reshape(bpj.shape)
if info != 0:
    print("Inversion algorithm did not converge to " + str(tol))

print("inversion time : " + str(time.time() - t))
