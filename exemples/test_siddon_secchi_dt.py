#!/usr/bin/env python
import os
import time
import numpy as np
import lo
import siddon
from siddon.solar import read_data
# data
obsrvtry = ('STEREO_A', 'STEREO_B')
data = siddon.solar.concatenate(
    [read_data(os.path.join(os.getenv('HOME'), 'data', 'siddon', '171dec08'), 
               bin_factor=4,
               obsrvtry=obs,
               time_window=['2008-12-01T00:00:00.000', 
                            '2008-12-15T00:00:00.000'],
               time_step= 4 * 3600.
               )
     for obs in obsrvtry])
data = siddon.solar.sort_data_array(data)
# scale A and B images
# the ratio of sensitivity between EUVI A and B
calibration_ba = {171:0.902023, 195:0.974536, 284:0.958269, 304:1.05954}
for i in xrange(data.shape[-1]):
    if data.header['OBSRVTRY'][i] == 'STEREO_B':
        data[..., i] /= calibration_ba[data.header['WAVELNTH'][i]]

# make sure it is 64 bits data
data.header['BITPIX'][:] = -64
# cube
shape = 3 * (128,)
header = {'CRPIX1':64.,
          'CRPIX2':64.,
          'CRPIX3':64.,
          'CDELT1':0.0234375,
          'CDELT2':0.0234375,
          'CDELT3':0.0234375,
          'CRVAL1':0.,
          'CRVAL2':0.,
          'CRVAL3':0.,}
cube = siddon.fa.zeros(shape, header=header)
# model
kwargs = {'obj_rmin':1., 'obj_rmax':1.4, 'data_rmax':1.3,
          'mask_negative':True, 'dt_min':100}
P, D, obj_mask, data_mask = siddon.models.stsrt(data, cube, **kwargs)
# apply mask to data
data *= (1 - data_mask)
# hyperparameters
hypers = (1e-1, 1e-1, 1e-1, 1e6)
# test time for one projection
t = time.time()
u = P.T * data.ravel()
print("maximal time : %f" % ((time.time() - t) * 100))
# inversion
t = time.time()
b = data.ravel()
#sol = lo.acg(P, b, D, hypers, maxiter=100)
sol = lo.rls(P, b, D, hypers, maxiter=100)
# reshape result
fsol = siddon.fa.asfitsarray(sol.reshape(obj_mask.shape), header=header)
print(time.time() - t)
fsol.tofits('stsrt_test.fits')
