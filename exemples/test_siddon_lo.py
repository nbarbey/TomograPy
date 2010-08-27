#!/bin/env python
import numpy as np
import os
import copy
import time
import siddon
import fitsarray as fa
# data 
path = os.path.join(os.getenv('HOME'), 'data', '171dec08')
obsrvtry = 'STEREO_A'
time_window = ['2008-12-01T00:00:00.000', '2008-12-03T00:00:00.000']
time_step = 4 * 3600. # one image every time_step seconds
data = siddon.secchi.read_data(path, bin_factor=4,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
# cube
shape = 3 * (128,)
header = dict()
for i in xrange(1, 4):
    header['CRPIX' + str(i)] = 64.
    header['CDELT' + str(i)] = 0.0234375
    header['CRVAL' + str(i)] = 0.
cube = fa.zeros(shape, header=header, dtype=np.float32)
P = siddon.siddon_sun_lo(data.header, cube.header)
t = time.time()
fbp = (P.T * data.flatten()).reshape(cube.shape)
print("backprojection time : " + str(time.time() - t))

t = time.time()
fbp0 = siddon.backprojector_sun(data, cube)
print("backprojection time : " + str(time.time() - t))

#assert np.all(fbp == fbp0)
