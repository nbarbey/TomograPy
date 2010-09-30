#!/usr/bin/env python
import numpy as np
import os
import copy
import time
import siddon
import lo
from lo.optimization import *
# data 
path = os.path.join(os.getenv('HOME'), 'data', '171dec08')
time_window = ['2008-12-01T00:00:00.000', '2008-12-15T00:00:00.000']
# one image every time_step seconds
time_step = 32 * 3600.
bin_factor = 16
obsrvtry = 'STEREO_A', 'STEREO_B'
data = siddon.secchi.read_data(path, bin_factor=bin_factor,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
# there is something wrong with the header !
data.header['BITPIX'] = -64 * np.ones(data.shape[-1])
# cube
shape = 3 * (128,)
header = {'CRPIX1':64., 'CRPIX2':64., 'CRPIX3':64.,
          'CDELT1':0.0234375, 'CDELT2':0.0234375, 'CDELT3':0.0234375,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
cube = siddon.fa.zeros(shape, header=header)
# masking
obj_mask = siddon.secchi.define_map_mask(cube, Rmin=1., Rmax=1.7)
Mo = lo.decimate(obj_mask)
data_mask = siddon.secchi.define_data_mask(data, Rmax=1.3, mask_negative=True)
Md = lo.decimate(data_mask)
# model
P = siddon.siddon_lo(data.header, cube.header, obstacle="sun")
D = [lo.diff(cube.shape, axis=i) for i in xrange(cube.ndim)]
hypers = cube.ndim * (1e-1, )
# apply mask model
P = Md * P * Mo.T
D = [Di * Mo.T for Di in D]
# inversion
t = time.time()
b = data[data_mask == 0]
sol = opt(P, b, D, hypers, maxiter=100)
fsol = cube.copy()
fsol[:] = (Mo.T * sol).reshape(fsol.shape)
print(time.time() - t)
