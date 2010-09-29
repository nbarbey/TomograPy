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
time_window = ['2008-12-01T00:00:00.000', '2008-12-15T00:00:00.000']
# one image every time_step seconds
time_step = 32 * 3600.
bin_factor = 16
obsrvtry = 'STEREO_A', 'STEREO_B'
data = siddon.secchi.read_data(path, bin_factor=bin_factor,
                               obsrvtry=obsrvtry,
                               time_window=time_window, 
                               time_step=time_step)
data.header['BITPIX'] = -64 * np.ones(data.shape[-1])
# cube
shape = 3 * (128,)
header = {'CRPIX1':64., 'CRPIX2':64., 'CRPIX3':64.,
          'CDELT1':0.0234375, 'CDELT2':0.0234375, 'CDELT3':0.0234375,
          'CRVAL1':0., 'CRVAL2':0., 'CRVAL3':0.,}
cube = fa.zeros(shape, header=header)
# masking
# map mask
Rmin = 1.
Rmax = 1.7
R = np.zeros(shape)
x, y, z = cube.axes()
X, Y = np.meshgrid(x, y)
for i, zt in enumerate(z):
    R[..., i] = np.sqrt(X ** 2 + Y ** 2 + zt ** 2)

obj_mask = (R < Rmin) + (R > Rmax)
Mo = lo.decimate(obj_mask)
# data mask
Rmax = 1.3
R = np.zeros(data.shape)
for i in xrange(data.shape[-1]):
    # get axes
    Rsun = data.header['RSUN'][i] / bin_factor
    crpix1 = data.header['CRPIX1'][i]
    x = (np.arange(data.shape[0]) - crpix1) / Rsun
    crpix2 = data.header['CRPIX2'][i]
    y = (np.arange(data.shape[0]) - crpix2) / Rsun
    X, Y = np.meshgrid(x, y)
    R[..., i] = np.sqrt(X ** 2 + Y ** 2)

data_mask = (R > Rmax)
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
#A = P.T * P + np.sum([h * d.T * d for h, d in zip(hypers, D)])
#b = P.T * data.flatten()
#callback = lo.iterative.CallbackFactory(verbose=True)
#x, info = spl.bicgstab(A, b, maxiter=100, callback=callback)
b = data[data_mask == 0]
x, info = lo.rls(P, b, D, hypers, maxiter=100,)
sol = cube.copy()
sol[:] = (Mo.T * x).reshape(cube.shape)
print(time.time() - t)
