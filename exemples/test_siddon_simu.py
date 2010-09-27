#!/usr/bin/env python
import time
import numpy as np
import scipy.sparse.linalg as spl
import siddon
# object
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':32, 'NAXIS2':32, 'NAXIS3':32,
          'CRPIX1':16., 'CRPIX2':16., 'CRPIX3':16.,
          'CDELT1':0.08, 'CDELT2':0.08, 'CDELT3':0.08,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
obj = siddon.simu.object_from_header(header)
obj[:] = siddon.phantom.shepp_logan(obj.shape)
#obj[:] = 1.
# data 
image_header = {'n_images':60,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':128, 'NAXIS2':128,
                'CRPIX1':64, 'CRPIX2':64,
                'CDELT1':12e-5, 'CDELT2':12e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
data = siddon.simu.circular_trajectory_data(**image_header)
data[:] = np.zeros(data.shape)
# projection
t = time.time()
data = siddon.projector(data, obj)
print("projection time : " + str(time.time() - t))
# backprojection
t = time.time()
x0 = obj.copy()
x0[:] = 0.
x0 = siddon.backprojector(data, x0.copy())
print("backprojection time : " + str(time.time() - t))
