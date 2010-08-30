#!/bin/env python
import time
import numpy as np
import siddon
from siddon.simu import circular_trajectory_data as get_data
from siddon.simu import spherical_object as get_object

# data 
image_header = {'n_images':16,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':512, 'NAXIS2':512, 
                'CRPIX1':256, 'CRPIX2':256,
                'CDELT1':3e-5, 'CDELT2':3e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
data = get_data(**image_header)
data[:] = np.zeros(data.shape)

# object
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':128, 'NAXIS2':128, 'NAXIS3':128,
          'CRPIX1':64., 'CRPIX2':64., 'CRPIX3':64.,
          'CDELT1':0.0234375, 'CDELT2':0.0234375, 'CDELT3':0.0234375,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
header['radius'] = 1.2
obj = get_object(**header)

# projector
P = siddon.siddon_lo(data.header, obj.header)

# projection
t = time.time()
data = siddon.projector(data, obj)
print("projection time : " + str(time.time() - t))

# backprojection
data1 = data.copy()
data1[:] = 1.
obj0 = obj.copy()
obj0[:] = 0.
t = time.time()
bpj = siddon.backprojector(data1, obj0)
print("projection time : " + str(time.time() - t))
