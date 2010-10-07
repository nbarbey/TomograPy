#!/usr/bin/env python
import os
import time
import lo
import siddon
from siddon.secchi import read_data
# data
obsrvtry = ('STEREO_A', 'STEREO_B')
data = siddon.secchi.concatenate(
    [read_data(os.path.join(os.getenv('HOME'), 'data', '171dec08'), 
               bin_factor=16,
               obsrvtry=obs,
               time_window=['2008-12-01T00:00:00.000', 
                            '2008-12-15T00:00:00.000'],
               time_step=32 * 3600.
               )
     for obs in obsrvtry])
data = siddon.secchi.sort_data_array(data)
# make sure it is 64 bits data
data.header['BITPIX'][:] = -64
# cube
shape = 3 * (64,)
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
kwargs = {'obj_rmin':1., 'obj_rmax':1.7, 'data_rmax':1.3,
          'mask_negative':True, 'dt_min':100}
P, D, obj_mask, data_mask, cube = siddon.models.stsrt(data, cube, **kwargs)
# hyperparameters
hypers = cube.ndim * (1e-1, )
# inversion
t = time.time()
b = data[data_mask == 0]
sol = lo.quadratic_optimization(P, b, D, hypers, maxiter=100)
# reshape result
fsol = siddon.fa.zeros(cube.shape, header=header)
fsol[obj_mask == 0] = sol.flatten()
print(time.time() - t)
