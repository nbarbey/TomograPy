#!/usr/bin/env python
import numpy as np
import os
import copy
import time
import siddon
import fitsarray as fa
import lo
import scipy.sparse.linalg as spl
# data 
path = os.path.join(os.getenv('HOME'), 'data', '171dec08')
obsrvtry = 'STEREO_A'
time_window = ['2008-12-01T00:00:00.000', '2008-12-03T00:00:00.000']
# one image every time_step seconds
time_step = 4 * 3600.
data = siddon.secchi.read_data(path, bin_factor=4,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
# cube
shape = 3 * (128,)
header = {'CRPIX1':64., 'CRPIX2':64., 'CRPIX3':64.,
          'CDELT1':0.0234375, 'CDELT2':0.0234375, 'CDELT3':0.0234375,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
cube = fa.zeros(shape, header=header, dtype=np.float32)
# model
P = siddon.siddon_lo(data.header, cube.header)
D = [lo.diff(cube.shape, axis=i, dtype=np.float32) for i in xrange(cube.ndim)]
hypers = cube.ndim * (1e0, )
hypers = [lo.mul(d.shape[1], h, dtype=np.float32) for h, d in zip(hypers, D)]
# inversion
t = time.time()
A = P.T * P + np.sum([h * d.T * d for h, d in zip(hypers, D)])
b = P.T * data.flatten()
callback = lo.iterative.CallbackFactory(verbose=True)
x, info = spl.bicgstab(A, b, maxiter=100, callback=callback)
#x, info = lo.acg(P, D, hypers, data.flatten(), maxiter=100,)
sol = cube.copy()
sol[:] = x.reshape(cube.shape)
print(time.time() - t)
