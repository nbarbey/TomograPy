#!/usr/bin/env python
import time
import numpy as np
import siddon
# object
n = 512
header = {'SIMPLE':True,'BITPIX':-64,
          'NAXIS1':n, 'NAXIS2':n, 'NAXIS3':n,
          'CRPIX1':n / 2., 'CRPIX2':n / 2., 'CRPIX3':n / 2.,
          'CDELT1':1.5 / n, 'CDELT2':1.5 / n, 'CDELT3':1.5 / n,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
obj = siddon.simu.object_from_header(header)
#obj[:] = siddon.phantom.shepp_logan(obj.shape)
obj[:] = 1.
# data
image_header = {'n_images':60,
                'SIMPLE':True, 'BITPIX':-64,
                'NAXIS1':128, 'NAXIS2':128,
                'CRPIX1':64, 'CRPIX2':64,
                'CDELT1':6e-5, 'CDELT2':6e-5,
                'CRVAL1':0., 'CRVAL2':0.,
                }
image_header['radius'] = 200.
for n_images in [10, 30, 60, 120]:
    print n_images
    image_header['n_images'] = n_images
    data = siddon.simu.circular_trajectory_data(**image_header)
    data[:] = np.zeros(data.shape)
    # projection
    t = time.time()
    data = siddon.projector(data, obj)
    print("projection time : " + str(time.time() - t))
    # backprojection
    x0 = siddon.fa.zeros(obj.shape, header=header)
    t = time.time()
    x0 = siddon.backprojector(data, x0)
    print("backprojection time : " + str(time.time() - t))

